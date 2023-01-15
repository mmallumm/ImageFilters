#include <opencv2/opencv.hpp>

class CV_EXPORTS TickMeter {
 public:
  TickMeter();
  void start();
  void stop();
  int64 getTimeTicks() const;
  double getTimeMicro() const;
  double getTimeMilli() const;
  double getTimeSec() const;
  int64 getCounter() const;
  void reset();

 private:
  int64 counter;
  int64 sumTime;
  int64 startTime;
};
std::ostream& operator<<(std::ostream& out, const TickMeter& tm);
