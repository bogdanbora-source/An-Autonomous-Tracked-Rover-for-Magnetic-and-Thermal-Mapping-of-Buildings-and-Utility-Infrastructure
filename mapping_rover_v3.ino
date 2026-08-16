/*
 * ============================================================
 *  MAGNETIC FIELD MAPPING ROVER — firmware v3
 *  Arduino Nano R4
 * ============================================================
 *
 *  WHAT'S NEW IN v3: STOP-AND-SAMPLE
 *  ---------------------------------
 *  v2 read the magnetometer while the motors were running. Motor
 *  current makes its own magnetic field, and that field changes
 *  with load and turn direction. No static calibration can cancel
 *  a field that varies in real time — so |B| ended up tracking the
 *  rover's HEADING instead of the ground. In the v2 logs |B| swung
 *  6x (282..1704) purely from driving/turning.
 *
 *  v3 does what real magnetometer surveys do:
 *      stop -> let motor currents die -> sample -> drive on
 *
 *  Costs ~0.5 s every 2 s. Buys you data you can actually trust.
 *
 *  Also new:
 *   - averages several magnetometer reads per sample (less noise)
 *   - logs mag_samples + was_moving so you can audit the data later
 *   - SELF-TEST MODE: spin the rover in place and verify that |B|
 *     stays constant. If it doesn't, the calibration is still bad
 *     and no map from this rover will mean anything. Set
 *     SELF_TEST_MODE = true to run it.
 *
 *  HARDWARE (unchanged)
 *  --------------------
 *      ENA->D5  IN1->D2  IN2->D3  IN3->D4  IN4->D7  ENB->D6
 *      HC-SR04: TRIG->D9  ECHO->D8      (NOTE: sensor was dead,
 *                                        this is the replacement)
 *      MicroSD: CS->D10  MOSI->D11  MISO->D12  SCK->D13
 *      QMC5883L: SDA->A4  SCL->A5   (on a mast, away from motors)
 *      NEO-6M GPS: TX->D0(RX)  RX->D1(TX)   [Serial1]
 *
 *  LED STATUS
 *  ----------
 *      fast continuous blink = SD FAILED, rover refuses to drive
 *      flash on each log row = logging OK
 *      solid                 = avoiding an obstacle
 *
 *  CSV COLUMNS
 *  -----------
 *  millis_ms,date,time,lat,lon,mag_x,mag_y,mag_z,mag_total,
 *  heading_deg,front_dist_cm,mag_samples,was_moving
 * ============================================================
 */

#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <TinyGPSPlus.h>
#include <QMC5883LCompass.h>

// ============================================================
//  SELF-TEST MODE
//  Set to true, upload, put the rover on the floor with clear
//  space, and it will slowly spin in place while logging.
//  Afterwards check mag_total in the CSV:
//     |B| roughly CONSTANT through the spin -> calibration good
//     |B| swinging 2x or more with heading  -> STILL BAD, redo it
//  Set back to false for real surveys.
// ============================================================
const bool SELF_TEST_MODE = true;

// ------------------------------------------------------------
//  PINS
// ------------------------------------------------------------
const int ENA = 5;
const int IN1 = 2;
const int IN2 = 3;
const int IN3 = 4;
const int IN4 = 7;
const int ENB = 6;

const int TRIG_PIN = 9;
const int ECHO_PIN = 8;
const int SD_CS    = 10;

// ------------------------------------------------------------
//  TUNING
// ------------------------------------------------------------
const int CRUISE_SPEED  = 200;
const int TURN_SPEED    = 190;
const int REVERSE_SPEED = 180;

const int OBSTACLE_CM = 20;
const int CAUTION_CM  = 40;

const unsigned long LOG_INTERVAL_MS = 2000;
const unsigned long REVERSE_MS      = 700;

// --- stop-and-sample timing ---
// After cutting the motors, the current in the windings (and the
// magnetic field it produces) takes a moment to collapse. Sampling
// too early defeats the whole purpose.
const unsigned long SETTLE_MS   = 350;   // wait after stopping
const int  MAG_SAMPLES          = 5;     // reads to average
const unsigned long MAG_GAP_MS  = 12;    // gap between reads

// Randomized turns (kills the hexagon from v1)
const unsigned long TURN_MIN_MS   = 450;
const unsigned long TURN_MAX_MS   = 1400;
const unsigned long ESCAPE_MIN_MS = 1400;
const unsigned long ESCAPE_MAX_MS = 2200;
const int DIR_FLIP_PERCENT = 45;

// Obstacle memory (needs a GPS fix -> outdoors only)
const int   MAX_OBSTACLES = 40;
const float CELL_SIZE_DEG = 0.00005;
const unsigned long STUCK_WINDOW_MS = 15000;
const int   STUCK_THRESHOLD = 3;

// Self-test
const unsigned long SELFTEST_SPIN_MS  = 700;   // spin a bit
const int           SELFTEST_STEPS    = 24;    // then sample; x24

// ------------------------------------------------------------
//  GLOBALS
// ------------------------------------------------------------
TinyGPSPlus     gps;
QMC5883LCompass compass;

bool sdOK = false;
char logFileName[16];

struct ObstacleCell { long latCell; long lonCell; };
ObstacleCell obstacles[MAX_OBSTACLES];
int obstacleCount = 0;
int obstacleWriteIdx = 0;

int lastTurnDir = 1;
unsigned long recentHits[STUCK_THRESHOLD];
int recentHitIdx = 0;

unsigned long lastLogMs = 0;

// NOTE: this struct MUST be declared before the first function in the
// file. The Arduino IDE auto-generates function prototypes and inserts
// them immediately before the first function definition — if the struct
// were declared later, those prototypes would reference an unknown type
// and you get: "'MagReading' does not name a type".
struct MagReading {
  float x, y, z, total;
  int   heading;
  int   samples;
};


// ============================================================
//  MOTORS
// ============================================================
void leftForward(int s)  { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);  analogWrite(ENA, s); }
void leftReverse(int s)  { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); analogWrite(ENA, s); }
void leftStop()          { digitalWrite(IN1, LOW);  digitalWrite(IN2, LOW);  analogWrite(ENA, 0); }

void rightForward(int s) { digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);  analogWrite(ENB, s); }
void rightReverse(int s) { digitalWrite(IN3, LOW);  digitalWrite(IN4, HIGH); analogWrite(ENB, s); }
void rightStop()         { digitalWrite(IN3, LOW);  digitalWrite(IN4, LOW);  analogWrite(ENB, 0); }

void driveForward(int s) { leftForward(s); rightForward(s); }
void driveReverse(int s) { leftReverse(s); rightReverse(s); }
void spinLeft(int s)     { leftReverse(s); rightForward(s); }
void spinRight(int s)    { leftForward(s); rightReverse(s); }
void stopMotors()        { leftStop();     rightStop();     }
void spin(int dir, int s){ if (dir > 0) spinRight(s); else spinLeft(s); }

// ============================================================
//  ULTRASONIC  (manual timing — pulseIn is unreliable on RA4M1)
// ============================================================
const unsigned long ECHO_TIMEOUT_US = 25000UL;

unsigned long readEchoMicros() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(5);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long t0 = micros();
  while (digitalRead(ECHO_PIN) == LOW) {
    if (micros() - t0 > ECHO_TIMEOUT_US) return 0;
  }
  unsigned long start = micros();
  while (digitalRead(ECHO_PIN) == HIGH) {
    if (micros() - start > ECHO_TIMEOUT_US) return 0;
  }
  return micros() - start;
}

int readDistanceCM() {
  unsigned long dur = readEchoMicros();
  if (dur == 0) return 999;
  int cm = (int)(dur / 58UL);
  if (cm < 2 || cm > 450) return 999;
  return cm;
}

int readDistanceFiltered() {
  int a = readDistanceCM(); delay(12);
  int b = readDistanceCM(); delay(12);
  int c = readDistanceCM();
  if (a > b) { int t = a; a = b; b = t; }
  if (b > c) { int t = b; b = c; c = t; }
  if (a > b) { int t = a; a = b; b = t; }
  return b;
}

// ============================================================
//  GPS
// ============================================================
void feedGPS() { while (Serial1.available() > 0) gps.encode(Serial1.read()); }

void feedGPSFor(unsigned long ms) {
  unsigned long t0 = millis();
  while (millis() - t0 < ms) feedGPS();
}

bool gpsHasFix() { return gps.location.isValid(); }

// ============================================================
//  MAGNETOMETER — averaged read
// ============================================================
MagReading readMagAveraged() {
  MagReading r;
  long sx = 0, sy = 0, sz = 0;
  for (int i = 0; i < MAG_SAMPLES; i++) {
    compass.read();
    sx += compass.getX();
    sy += compass.getY();
    sz += compass.getZ();
    if (i < MAG_SAMPLES - 1) { feedGPSFor(MAG_GAP_MS); }
  }
  r.x = (float)sx / MAG_SAMPLES;
  r.y = (float)sy / MAG_SAMPLES;
  r.z = (float)sz / MAG_SAMPLES;
  r.total = sqrt(r.x*r.x + r.y*r.y + r.z*r.z);
  r.heading = compass.getAzimuth();
  r.samples = MAG_SAMPLES;
  return r;
}

// ============================================================
//  OBSTACLE MEMORY
// ============================================================
long toCell(double deg) { return (long)floor(deg / CELL_SIZE_DEG); }

bool isKnownObstacleCell(long latC, long lonC) {
  for (int i = 0; i < obstacleCount; i++)
    if (obstacles[i].latCell == latC && obstacles[i].lonCell == lonC) return true;
  return false;
}

void rememberObstacleHere() {
  if (!gpsHasFix()) return;
  long latC = toCell(gps.location.lat());
  long lonC = toCell(gps.location.lng());
  if (isKnownObstacleCell(latC, lonC)) return;
  obstacles[obstacleWriteIdx].latCell = latC;
  obstacles[obstacleWriteIdx].lonCell = lonC;
  obstacleWriteIdx = (obstacleWriteIdx + 1) % MAX_OBSTACLES;
  if (obstacleCount < MAX_OBSTACLES) obstacleCount++;
}

bool headingIntoKnownObstacle() {
  if (!gpsHasFix() || !gps.course.isValid()) return false;
  double hr = gps.course.deg() * DEG_TO_RAD;
  double dLat = cos(hr) * (CELL_SIZE_DEG * 0.8);
  double dLon = sin(hr) * (CELL_SIZE_DEG * 0.8);
  return isKnownObstacleCell(toCell(gps.location.lat() + dLat),
                             toCell(gps.location.lng() + dLon));
}

bool registerHitAndCheckStuck() {
  recentHits[recentHitIdx] = millis();
  recentHitIdx = (recentHitIdx + 1) % STUCK_THRESHOLD;
  unsigned long now = millis();
  for (int i = 0; i < STUCK_THRESHOLD; i++) {
    if (recentHits[i] == 0) return false;
    if (now - recentHits[i] > STUCK_WINDOW_MS) return false;
  }
  return true;
}

// ============================================================
//  SD LOGGING
// ============================================================
void pickLogFileName() {
  for (int i = 0; i < 1000; i++) {
    snprintf(logFileName, sizeof(logFileName), "LOG%03d.CSV", i);
    if (!SD.exists(logFileName)) return;
  }
}

void writeHeader() {
  File f = SD.open(logFileName, FILE_WRITE);
  if (!f) { sdOK = false; return; }
  f.println(F("millis_ms,date,time,lat,lon,mag_x,mag_y,mag_z,mag_total,"
              "heading_deg,front_dist_cm,mag_samples,was_moving"));
  f.close();
}

void writeRow(const MagReading &m, int frontDist, bool wasMoving) {
  if (!sdOK) return;

  File f = SD.open(logFileName, FILE_WRITE);
  if (!f) { sdOK = false; return; }

  f.print(millis()); f.print(',');

  if (gps.date.isValid() && gps.time.isValid()) {
    char buf[24];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d,%02d:%02d:%02d",
             gps.date.year(), gps.date.month(), gps.date.day(),
             gps.time.hour(), gps.time.minute(), gps.time.second());
    f.print(buf);
  } else {
    f.print(',');                       // empty date + empty time
  }
  f.print(',');

  if (gpsHasFix()) {
    f.print(gps.location.lat(), 6); f.print(',');
    f.print(gps.location.lng(), 6); f.print(',');
  } else {
    f.print(F(",,"));
  }

  f.print(m.x, 1);     f.print(',');
  f.print(m.y, 1);     f.print(',');
  f.print(m.z, 1);     f.print(',');
  f.print(m.total, 1); f.print(',');
  f.print(m.heading);  f.print(',');
  f.print(frontDist);  f.print(',');
  f.print(m.samples);  f.print(',');
  f.println(wasMoving ? 1 : 0);

  f.close();

  digitalWrite(LED_BUILTIN, HIGH);
  delay(25);
  digitalWrite(LED_BUILTIN, LOW);
}

// ------------------------------------------------------------
//  THE CORE OF v3: stop, settle, sample, resume
// ------------------------------------------------------------
void stopAndSample(int frontDist) {
  stopMotors();                 // 1. cut the motors
  feedGPSFor(SETTLE_MS);        // 2. let the winding currents (and
                                //    their magnetic field) collapse
  MagReading m = readMagAveraged();   // 3. sample a quiet sensor
  writeRow(m, frontDist, false);      // 4. log it (was_moving = 0)
  // caller resumes driving
}

// ============================================================
//  AVOIDANCE
// ============================================================
void avoidObstacle(bool remembered) {
  digitalWrite(LED_BUILTIN, HIGH);
  stopMotors();
  feedGPSFor(120);

  if (!remembered) {
    rememberObstacleHere();
    driveReverse(REVERSE_SPEED);
    feedGPSFor(REVERSE_MS);
    stopMotors();
    feedGPSFor(100);
  }

  if (random(100) < DIR_FLIP_PERCENT) lastTurnDir = -lastTurnDir;

  if (registerHitAndCheckStuck()) {
    spin(lastTurnDir, TURN_SPEED);
    feedGPSFor(random(ESCAPE_MIN_MS, ESCAPE_MAX_MS));
    stopMotors();
    for (int i = 0; i < STUCK_THRESHOLD; i++) recentHits[i] = 0;
  } else {
    spin(lastTurnDir, TURN_SPEED);
    feedGPSFor(random(TURN_MIN_MS, TURN_MAX_MS));
    stopMotors();
    feedGPSFor(100);

    if (readDistanceFiltered() < OBSTACLE_CM + 10) {
      lastTurnDir = -lastTurnDir;
      spin(lastTurnDir, TURN_SPEED);
      feedGPSFor(random(TURN_MIN_MS, TURN_MAX_MS) * 2);
      stopMotors();
    }
  }
  digitalWrite(LED_BUILTIN, LOW);
}

// ============================================================
//  SELF-TEST: spin in place, sample at each step.
//  Afterwards, plot mag_total against heading_deg. A well
//  calibrated sensor gives a FLAT line. A bad one gives a wave.
// ============================================================
void runSelfTest() {
  Serial.println(F("SELF-TEST: spinning and sampling..."));
  for (int i = 0; i < SELFTEST_STEPS; i++) {
    spinRight(TURN_SPEED);
    feedGPSFor(SELFTEST_SPIN_MS);
    stopMotors();
    feedGPSFor(SETTLE_MS);

    MagReading m = readMagAveraged();
    writeRow(m, 999, false);

    Serial.print(F("  step ")); Serial.print(i + 1);
    Serial.print(F("  heading ")); Serial.print(m.heading);
    Serial.print(F("  |B| "));     Serial.println(m.total, 0);
  }
  stopMotors();
  Serial.println(F("SELF-TEST DONE."));
  Serial.println(F("Check |B| above: roughly constant = GOOD."));
  Serial.println(F("Swinging 2x or more with heading = RECALIBRATE."));
  while (true) {                       // halt; flash slowly
    digitalWrite(LED_BUILTIN, HIGH); delay(600);
    digitalWrite(LED_BUILTIN, LOW);  delay(600);
  }
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  stopMotors();

  Serial.begin(115200);
  Serial1.begin(9600);
  Wire.begin();

  randomSeed(analogRead(A0) ^ micros());

  delay(2000);
  Serial.println(F("=== Mapping Rover v3 ==="));

  compass.init();
  // Calibration done WITH the sensor mounted on the rover, tilted
  // through all orientations. Scales all near 1.0 = healthy.
  compass.setCalibrationOffsets(-717.00, -827.00, -411.00);
  compass.setCalibrationScales(1.22, 0.96, 0.88);
  Serial.println(F("QMC5883L calibrated"));

  sdOK = SD.begin(SD_CS);
  if (sdOK) {
    pickLogFileName();
    writeHeader();
    Serial.print(F("SD OK -> ")); Serial.println(logFileName);
  } else {
    Serial.println(F("!!! SD FAILED !!!"));
  }

  for (int i = 0; i < STUCK_THRESHOLD; i++) recentHits[i] = 0;

  delay(1000);

  if (SELF_TEST_MODE) {
    if (!sdOK) Serial.println(F("(self-test running, but not logging!)"));
    runSelfTest();      // never returns
  }

  Serial.println(F("=== Roaming (stop-and-sample) ==="));
}

// ============================================================
//  MAIN LOOP
// ============================================================
void loop() {
  // SD dead? Refuse to drive — don't wander around collecting nothing.
  if (!sdOK) {
    stopMotors();
    digitalWrite(LED_BUILTIN, HIGH); delay(80);
    digitalWrite(LED_BUILTIN, LOW);  delay(80);
    return;
  }

  feedGPS();

  int dist = readDistanceFiltered();

  // ---- sample on schedule, ALWAYS, with the motors stopped ----
  if (millis() - lastLogMs >= LOG_INTERVAL_MS) {
    lastLogMs = millis();
    stopAndSample(dist);
    // re-read distance: we've been stationary for ~0.4 s and the
    // world may have changed (or we stopped facing a wall)
    dist = readDistanceFiltered();
  }

  // ---- then drive ----
  if (dist < OBSTACLE_CM) {
    avoidObstacle(false);
    return;
  }

  if (headingIntoKnownObstacle()) {
    avoidObstacle(true);
    return;
  }

  driveForward(dist < CAUTION_CM ? CRUISE_SPEED - 40 : CRUISE_SPEED);
}
