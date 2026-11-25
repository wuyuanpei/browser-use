"""Prepare the viewport image and project CSS rectangles into its JPEG pixels."""

import base64
import io
import math
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class PreparedImage:
	jpeg_b64: str
	width: int
	height: int
	scale: float  # captured pixels per CSS pixel
	bytes: int
	rect_scale: tuple[float, float] = (1.0, 1.0)  # JPEG pixels per CSS pixel, including size rounding

	def project_rect(self, rect: dict[str, int]) -> dict[str, int]:
		if self.rect_scale == (1.0, 1.0):
			return rect
		scale_x, scale_y = self.rect_scale
		x = math.floor(rect['x'] * scale_x)
		y = math.floor(rect['y'] * scale_y)
		right = min(self.width, math.ceil((rect['x'] + rect['w']) * scale_x))
		bottom = min(self.height, math.ceil((rect['y'] + rect['h']) * scale_y))
		return {'x': x, 'y': y, 'w': right - x, 'h': bottom - y}


def prepare_screenshot(
	png_b64: str | None, viewport_w: int, viewport_h: int, *, max_side: int, quality: int
) -> PreparedImage | None:
	"""Normalize a BrowserStateSummary capture to its CSS viewport, optionally capping the long side."""
	if not png_b64 or min(viewport_w, viewport_h, max_side) <= 0:
		return None
	try:
		with Image.open(io.BytesIO(base64.b64decode(png_b64, validate=True)), formats=['PNG']) as source:
			scale = source.width / viewport_w
			rect_scale = min(1.0, max_side / max(viewport_w, viewport_h))
			width = max(1, round(viewport_w * rect_scale))
			height = max(1, round(viewport_h * rect_scale))
			image = source.convert('RGB')
			if image.size != (width, height):
				image = image.resize((width, height), Image.Resampling.LANCZOS)
			buffer = io.BytesIO()
			image.save(buffer, format='JPEG', quality=quality)
		jpeg = buffer.getvalue()
		return PreparedImage(
			base64.b64encode(jpeg).decode('ascii'), width, height, scale, len(jpeg), (width / viewport_w, height / viewport_h)
		)
	except Exception:
		# A missing or unreadable capture must leave the text-only decision available.
		return None
