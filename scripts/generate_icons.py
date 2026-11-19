from PIL import Image, ImageDraw
import math
from pathlib import Path

out_dir = Path(__file__).resolve().parents[1] / "frontend" / "public"

def draw_icon(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Background rounded rect
    rx = int(size * 0.22)
    draw.rounded_rectangle([0, 0, size, size], radius=rx, fill=(14, 27, 44, 255))

    # Outer meter dial
    center = size / 2
    radius = size * 0.32
    width = max(2, int(size * 0.05))
    draw.arc(
        [center - radius, center - radius, center + radius, center + radius],
        start=135,
        end=405,
        fill=(197, 208, 222, 160),
        width=width,
    )

    # Needle (accent color #eb6834 = 235, 104, 52)
    needle_len = radius * 0.85
    angle_rad = math.radians(315)  # pointing up-right
    end_x = center + needle_len * math.cos(angle_rad)
    end_y = center + needle_len * math.sin(angle_rad)
    needle_width = max(3, int(size * 0.065))
    draw.line([center, center, end_x, end_y], fill=(235, 104, 52, 255), width=needle_width)

    # Center pivot
    pivot_r = max(3, int(size * 0.07))
    draw.ellipse([center - pivot_r, center - pivot_r, center + pivot_r, center + pivot_r], fill=(197, 208, 222, 255))

    return img

if __name__ == "__main__":
    draw_icon(192).save(out_dir / "pwa-192x192.png")
    draw_icon(512).save(out_dir / "pwa-512x512.png")
    draw_icon(180).save(out_dir / "apple-touch-icon.png")
    print("PWA icons generated successfully in", out_dir)
