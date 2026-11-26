import base64
import io

import pytest
from PIL import Image

from browser_use.agent.jev.vision import prepare_screenshot


def png(width: int, height: int) -> str:
	buffer = io.BytesIO()
	Image.new('RGBA', (width, height), (42, 145, 96, 255)).save(buffer, format='PNG')
	return base64.b64encode(buffer.getvalue()).decode('ascii')


@pytest.mark.parametrize('dpr', [1, 1.5, 2])
def test_prepare_normalizes_to_css_pixels(dpr):
	image = prepare_screenshot(png(int(400 * dpr), int(250 * dpr)), 400, 250, max_side=1600, quality=72)
	assert image is not None
	assert (image.width, image.height, image.scale, image.rect_scale) == (400, 250, dpr, (1, 1))
	data = base64.b64decode(image.jpeg_b64, validate=True)
	assert len(data) == image.bytes
	with Image.open(io.BytesIO(data)) as jpeg:
		assert (jpeg.format, jpeg.mode, jpeg.size) == ('JPEG', 'RGB', (400, 250))


@pytest.mark.parametrize('input_data', [None, '', 'not base64', 'Y29ycnVwdA=='])
def test_unreadable_screenshot(input_data):
	assert prepare_screenshot(input_data, 640, 480, max_side=1600, quality=72) is None


@pytest.mark.parametrize('css', [(401, 1000), (1000, 401)])
def test_capped_size_and_rect_projection(css):
	w, h = css
	image = prepare_screenshot(png(w * 2, h * 2), w, h, max_side=320, quality=72)
	assert image is not None and image.scale == 2
	assert (image.width, image.height) == (round(w * 0.32), round(h * 0.32))
	assert image.project_rect({'x': 0, 'y': 0, 'w': w, 'h': h}) == {'x': 0, 'y': 0, 'w': image.width, 'h': image.height}
	assert image.project_rect({'x': w - 1, 'y': h - 1, 'w': 1, 'h': 1}) == {
		'x': image.width - 1,
		'y': image.height - 1,
		'w': 1,
		'h': 1,
	}
	assert prepare_screenshot(png(w, h), 0, h, max_side=320, quality=72) is None
