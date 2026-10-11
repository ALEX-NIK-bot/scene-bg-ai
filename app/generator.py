import base64
import io
import os
from typing import Dict, Any, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

try:
    import torch
    from diffusers import AutoPipelineForText2Image
except Exception:  # pragma: no cover
    torch = None
    AutoPipelineForText2Image = None


class SceneBackgroundGenerator:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0"):
        self.model_id = model_id
        self.pipeline = None

    def _build_prompt(self, prompt: str, style: str, mood: str, quality: int, detail: int, saturation: int) -> str:
        style_map = {
            "realistic": "realistic cinematic scene",
            "anime": "anime illustration",
            "cinematic": "cinematic composition",
            "watercolor": "watercolor painting",
            "oil": "oil painting",
            "concept": "concept art",
            "3d": "3D render",
        }
        mood_map = {
            "peaceful": "peaceful calm atmosphere",
            "dark": "dark moody atmosphere",
            "mysterious": "mysterious hidden atmosphere",
            "dramatic": "dramatic tension",
            "dreamy": "dreamy romantic atmosphere",
            "epic": "epic heroic atmosphere",
        }

        style_text = style_map.get(style, "cinematic scene")
        mood_text = mood_map.get(mood, "cinematic atmosphere")
        quality_text = "high quality, polished" if quality >= 6 else "good quality"
        detail_text = "detailed environment" if detail >= 6 else "clean environment"
        saturation_text = "rich saturated colors" if saturation >= 6 else "balanced colors"
        return f"{prompt}, {style_text}, {mood_text}, {quality_text}, {detail_text}, {saturation_text}, ultra detailed background, wide shot, depth of field" 

    def _load_sd_pipeline(self):
        if AutoPipelineForText2Image is None or torch is None:
            return None

        try:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None,
            )
            if torch.cuda.is_available():
                pipeline.to("cuda")
            else:
                pipeline.to("cpu")
            return pipeline
        except Exception:
            return None

    def _render_fallback(self, prompt: str, style: str, mood: str, width: int, height: int, quality: int, detail: int, saturation: int) -> Image.Image:
        image = Image.new("RGB", (width, height), color=(12, 18, 30))
        draw = ImageDraw.Draw(image)

        palette = {
            "realistic": [(25, 35, 50), (70, 120, 50), (180, 180, 110)],
            "anime": [(52, 31, 94), (168, 85, 247), (253, 164, 175)],
            "cinematic": [(15, 23, 42), (37, 99, 235), (248, 113, 113)],
            "watercolor": [(125, 211, 252), (192, 132, 252), (254, 240, 138)],
            "oil": [(120, 53, 15), (251, 146, 60), (252, 165, 165)],
            "concept": [(15, 23, 42), (14, 165, 233), (244, 114, 182)],
            "3d": [(17, 24, 39), (14, 165, 233), (45, 212, 191)],
        }
        mood_palette = {
            "peaceful": [(147, 197, 253), (187, 247, 208), (255, 255, 255)],
            "dark": [(2, 6, 23), (17, 24, 39), (55, 65, 81)],
            "mysterious": [(15, 23, 42), (49, 46, 129), (76, 29, 149)],
            "dramatic": [(17, 24, 39), (185, 28, 28), (251, 146, 60)],
            "dreamy": [(216, 180, 254), (167, 139, 250), (249, 168, 212)],
            "epic": [(120, 53, 15), (251, 146, 60), (253, 224, 71)],
        }

        grad_a, grad_b, grad_c = palette.get(style, palette["cinematic"])
        m1, m2, m3 = mood_palette.get(mood, mood_palette["peaceful"])

        # gradient sky
        for y in range(height):
            t = y / height
            r = int(grad_a[0] * (1 - t) + grad_b[0] * t)
            g = int(grad_a[1] * (1 - t) + grad_b[1] * t)
            b = int(grad_a[2] * (1 - t) + grad_b[2] * t)
            for x in range(width):
                draw.point((x, y), (r, g, b))

        # soft lights
        for i in range(20):
            x = (i * 73) % width
            y = (i * 53) % (height // 2)
            radius = 150 + (i % 5) * 35
            glow = Image.new("RGBA", (radius * 2, radius * 2), (255, 255, 255, 0))
            glow_draw = ImageDraw.Draw(glow)
            glow_draw.ellipse((0, 0, radius * 2, radius * 2), fill=(255, 255, 255, 24))
            img = image.copy()
            pasted = Image.alpha_composite(img.convert("RGBA"), glow.crop((0, 0, radius * 2, radius * 2)).resize((radius * 2, radius * 2)))
            image = pasted.convert("RGB")

        # silhouettes
        silhouette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        sid = ImageDraw.Draw(silhouette)
        sid.polygon([(0, height), (0, int(height * 0.7)), (width * 0.18, int(height * 0.62)), (width * 0.33, height)], fill=(20, 20, 30, 180))
        sid.polygon([(width * 0.22, height), (width * 0.40, int(height * 0.55)), (width * 0.65, height)], fill=(18, 24, 36, 200))
        sid.polygon([(width * 0.5, height), (width * 0.7, int(height * 0.58)), (width * 0.95, height)], fill=(10, 12, 20, 220))
        image = Image.alpha_composite(image.convert("RGBA"), silhouette).convert("RGB")

        # vignette
        vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        vdraw = ImageDraw.Draw(vignette)
        for i in range(200):
            alpha = int(110 - i * 0.4)
            if alpha < 0:
                alpha = 0
            vdraw.rectangle((i, 0, width - i, height), outline=(0, 0, 0, alpha))
        image = Image.alpha_composite(image.convert("RGBA"), vignette).convert("RGB")

        # text overlay
        text = prompt[:54]
        text_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        tdraw = ImageDraw.Draw(text_image)
        tdraw.text((width // 2, height // 2 - 20), "Scene BG AI", fill=(255, 255, 255, 220), anchor="mm", font=None)
        tdraw.text((width // 2, height // 2 + 25), text, fill=(255, 255, 255, 200), anchor="mm")
        image = Image.alpha_composite(image.convert("RGBA"), text_image).convert("RGB")

        # mild blur/saturation for quality
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.0 + saturation / 15)

        if detail >= 7:
            image = image.filter(ImageFilter.GaussianBlur(radius=0.3))

        return image

    def generate(self, prompt: str, style: str, mood: str, quality: int = 7, detail: int = 7, saturation: int = 7, width: int = 1280, height: int = 720) -> Tuple[str, str]:
        if not prompt or not prompt.strip():
            raise ValueError("prompt is required")

        final_prompt = self._build_prompt(prompt, style, mood, quality, detail, saturation)

        pipeline = self.pipeline or self._load_sd_pipeline()
        if pipeline is not None:
            try:
                result = pipeline(
                    prompt=final_prompt,
                    negative_prompt="blurry, low quality, duplicate, watermark, text, logo, noisy background, bad composition",
                    width=width,
                    height=height,
                    num_inference_steps=max(18, 25 + quality),
                    guidance_scale=7.5,
                )
                image = result.images[0]
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                return f"data:image/png;base64,{encoded}", "success"
            except Exception:
                # fallback to fallback render
                pass

        image = self._render_fallback(prompt, style, mood, width, height, quality, detail, saturation)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}", "fallback"
