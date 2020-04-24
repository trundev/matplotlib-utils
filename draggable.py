import numpy
import matplotlib.pyplot as pyplot

HOVER_EFFECTS = True

class Draggable(object):
    def __init__(self, figure, on_moved=None):
        self.on_moved = on_moved
        self._figure = figure
        self._actor = None
        # Drag context
        self._drag_base = None
        self._drag_direction = None
        self._drag_contains = None
        self._drag_offset = None
        # Hover context
        self._hovered = None
        self._actor_color = None
        # Catch mouse events
        self._figure.canvas.mpl_connect('button_press_event', self._on_click)
        self._figure.canvas.mpl_connect('button_release_event', self._on_release)
        self._figure.canvas.mpl_connect('motion_notify_event', self._on_motion)

    def update_actor(self, actor):
        """Update actor/collection with a new version, in response to on_moved())"""
        # The actor should be already removed
        if self._actor and self._actor.figure:
            self._actor.remove()
        self._actor = actor
        self._actor_color = actor.get_edgecolor()

    def _update_plot(self):
        self._figure.canvas.draw_idle()

    def _on_click(self, event):
        """Mouse click event callback"""
        # MouseButton.MIDDLE
        if event.button == 2:
            res = self._actor.contains(event)
            if res and res[0]:
                _, contains = res
                self._drag_base = numpy.array([event.x, event.y])
                self._drag_contains = contains
                self._drag_offset = self._actor.get_offsets()
                # Single direction only
                main_path = self._actor.get_paths()[0]
                self._drag_direction = main_path.vertices[0] - main_path.vertices[1]
        return False

    def _on_release(self, event):
        """Mouse release event callback"""
        if self._drag_base is not None:
            if self.on_moved is not None and event.inaxes is self._actor.axes:
                offset = self._actor.get_offsets() - self._drag_offset
                offset = offset[0]      # Support single offset only
                # Convert offset into the drag-direction vector units
                if self._drag_direction is not None:
                    offset = self._drag_direction.dot(offset)
                self.on_moved(self, offset)
            self._drag_base = None

    def _on_motion(self, event):
        """Mouse motion event callback"""
        if self._drag_base is not None:
            if event.inaxes is self._actor.axes:
                offset = numpy.array([event.x, event.y]) - self._drag_base
                if self._drag_direction is not None:
                    offset = self._drag_direction * self._drag_direction.dot(offset)
                    offset /= self._drag_direction.dot(self._drag_direction)
                self._actor.set_offsets(offset + self._drag_offset)
            else:
                self._actor.set_offsets(self._drag_offset)
            self._update_plot()

        elif HOVER_EFFECTS:
            res = self._actor.contains(event)
            if res and res[0]:
                _, contains = res
                self._hovered = contains
                color = self._actor_color.copy()
                color[...,3] = 1
                self._actor.set_edgecolor(color)
            elif self._hovered is not None:
                self._actor.set_edgecolor(self._actor_color)
                self._hovered = None
            self._update_plot()
