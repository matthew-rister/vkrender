#ifndef SRC_GRAPHICS_DELTA_TIME_H_
#define SRC_GRAPHICS_DELTA_TIME_H_

#include <chrono>

namespace qem {

/** \brief Calculates the time between frames. */
class DeltaTime {
  using Clock = std::chrono::steady_clock;
  using Duration = std::chrono::duration<float>;
  using TimePoint = std::chrono::time_point<Clock, Duration>;

public:
  DeltaTime() noexcept : current_time_{Clock::now()}, previous_time_{current_time_} {}

  /**
   * \brief Gets the current delta time.
   * \return The time in float seconds since <tt>DeltaTime::Update</tt> was called.
   */
  [[nodiscard]] Duration::rep get() const noexcept { return delta_time_.count(); }

  /** \brief Calculates the current delta time. This should be called each frame in the main render loop. */
  void Update() noexcept {
    delta_time_ = current_time_ - previous_time_;
    previous_time_ = current_time_;
    current_time_ = Clock::now();
  }

private:
  TimePoint current_time_, previous_time_;
  Duration delta_time_{};
};

}  // namespace qem

#endif  // SRC_GRAPHICS_DELTA_TIME_H_
