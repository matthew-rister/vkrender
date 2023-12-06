#ifndef SRC_GRAPHICS_WINDOW_H_
#define SRC_GRAPHICS_WINDOW_H_

#include <concepts>
#include <functional>
#include <utility>

#include <GLFW/glfw3.h>
#include <glm/vec2.hpp>

namespace qem {

/** \brief An abstraction for a GLFW window. */
class Window {
public:
  /**
   * \brief Initializes a window.
   * \param title The window title.
   * \param window_dimensions The window width and height.
   * \param opengl_version The OpenGL major and minor version.
   */
  Window(const char* title, const std::pair<int, int>& window_dimensions, const std::pair<int, int>& opengl_version);

  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;

  Window(Window&&) noexcept = delete;
  Window& operator=(Window&&) noexcept = delete;

  ~Window() noexcept;

  /**
   * \brief Sets a callback to be invoked when a discrete key press is detected.
   * \param on_key_press The callback to be invoked on key press parameterized by the active key code.
   */
  void OnKeyPress(std::invocable<int> auto&& on_key_press) {
    on_key_press_ = std::forward<decltype(on_key_press_)>(on_key_press);
  }

  /**
   * \brief Sets a callback to be invoked when a scroll event is detected.
   * \param on_scroll A callback to be invoked on scroll parameterized by x/y scroll offsets (respectively).
   */
  void OnScroll(std::invocable<double, double> auto&& on_scroll) {
    on_scroll_ = std::forward<decltype(on_scroll_)>(on_scroll);
  }

  /**
   * \brief Gets the window dimensions.
   * \return A pair representing the window's width and height.
   */
  [[nodiscard]] std::pair<int, int> GetSize() const noexcept {
    int width{}, height{};
    glfwGetWindowSize(window_, &width, &height);
    return std::pair{width, height};
  }

  /**
   * \brief Gets the windows aspect ratio.
   * \return The ratio of the windows width to its height (e.g., 16/9).
   */
  [[nodiscard]] float GetAspectRatio() const noexcept {
    const auto [width, height] = GetSize();
    return height > 0 ? static_cast<float>(width) / static_cast<float>(height) : 0;
  }

  /**
   * \brief Gets the cursor position.
   * \return The (x,y) coordinates of the cursor position in the window.
   */
  [[nodiscard]] glm::dvec2 GetCursorPosition() const noexcept {
    double x{}, y{};
    glfwGetCursorPos(window_, &x, &y);
    return glm::dvec2{x, y};
  }

  /** \brief Sets the window title. */
  void SetTitle(const char* const title) const noexcept { glfwSetWindowTitle(window_, title); }

  /**
   * \brief Determines if the window is closed.
   * \return \c true if the window is closed, otherwise \c false.
   */
  [[nodiscard]] bool IsClosed() const noexcept { return glfwWindowShouldClose(window_) == GLFW_TRUE; }

  /**
   * \brief Determines if a key is pressed.
   * \param key_code The key code to evaluate (e.g., GLFW_KEY_S).
   * \return \c true if \p key is pressed, otherwise \c false.
   */
  [[nodiscard]] bool IsKeyPressed(const int key_code) const noexcept {
    return glfwGetKey(window_, key_code) == GLFW_PRESS;
  }

  /**
   * \brief Determines if a mouse button is pressed.
   * \param button_code The mouse button code (e.g., GLFW_MOUSE_BUTTON_LEFT).
   * \return \c true if \p button is pressed, otherwise \c false.
   */
  [[nodiscard]] bool IsMouseButtonPressed(const int button_code) const noexcept {
    return glfwGetMouseButton(window_, button_code) == GLFW_PRESS;
  }

  /** \brief Updates the window for the next frame in the main render loop. */
  void Update() const noexcept {
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }

private:
  GLFWwindow* window_ = nullptr;
  std::function<void(int)> on_key_press_;
  std::function<void(double, double)> on_scroll_;
};

}  // namespace qem

#endif  // SRC_GRAPHICS_WINDOW_H_
