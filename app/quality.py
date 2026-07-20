"""
Görüntü kalite kontrolü modülü.
- Bulanıklık tespiti (Laplacian varyans yöntemi)
- Yüz tespit güven skoru kontrolü
"""

import cv2
import numpy as np


def check_blur(image: np.ndarray, threshold: float = 50.0) -> tuple[bool, float]:
    """
    OpenCV Laplacian varyans yöntemiyle görüntünün bulanıklığını ölçer.
    
    Parametreler:
        image:     OpenCV ile okunmuş BGR görüntü (numpy array)
        threshold: Altında kalırsa bulanık kabul edilecek eşik değeri.
                   Değer ne kadar düşükse görüntü o kadar bulanıktır.
                   Önerilen: 50–100 arası.
    
    Dönüş:
        (is_blurry: bool, variance: float)
        variance küçükse (threshold altında) görüntü bulanıktır.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return (variance < threshold), variance
