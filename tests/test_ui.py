"""UI 组件单元测试。"""

from __future__ import annotations

import queue
import unittest

from sensor_noise.ui.app import MainWindow


class TestPlotController(unittest.TestCase):
    def test_current_tab_on_python313_notebook(self) -> None:
        app = MainWindow()
        try:
            tab = app.plots.current_tab()
            self.assertIn(tab, app.plots.TABS)
        finally:
            app.destroy()

    def test_task_queue_handles_analyze_messages(self) -> None:
        app = MainWindow()
        try:
            app._handle_task(("analyze_progress", 2, 8, "ch2"))
            app._handle_task(("analyzed_ok",))
        finally:
            app.destroy()

    def test_task_queue_processes_one_message_per_poll(self) -> None:
        app = MainWindow()
        try:
            app._task_queue.put(("analyze_progress", 1, 8, "ch1"))
            app._task_queue.put(("analyze_progress", 2, 8, "ch2"))
            app._poll_task_queue()
            self.assertEqual(app._task_queue.qsize(), 1)
        finally:
            app.destroy()

    def test_task_queue_is_thread_safe_queue(self) -> None:
        app = MainWindow()
        try:
            self.assertIsInstance(app._task_queue, queue.Queue)
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
