#!/usr/bin/env python3
"""
Olympic CamGuard - 改進的 AI 圖像分析模組
整合多種檢測方法提高準確度
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """檢測結果數據類"""
    feature_type: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    description: str = ""


class AdvancedVenueDetector:
    """進階場館檢測器"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

        # 預定義的場館特徵
        self.venue_features = {
            "olympic_rings": self._detect_olympic_rings,
            "stadium_structure": self._detect_stadium_structure,
            "sports_equipment": self._detect_sports_equipment,
            "venue_signage": self._detect_text_signs,
            "architectural_features": self._detect_architecture,
            "crowd_patterns": self._detect_crowd_patterns,
        }

        # 色彩特徵定義
        self.olympic_colors = {
            "blue": [(100, 150, 50), (130, 255, 255)],
            "yellow": [(20, 100, 100), (30, 255, 255)],
            "green": [(50, 100, 50), (70, 255, 255)],
            "red": [(0, 120, 70), (10, 255, 255)],
            "black": [(0, 0, 0), (180, 255, 30)]
        }

    async def analyze_image_advanced(self, image_data: bytes) -> Dict:
        """進階圖像分析主函數"""
        try:
            # 解碼圖像
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("無法解析圖像")

            # 預處理
            img_processed = self._preprocess_image(img)

            # 並行執行多種檢測
            detection_tasks = []
            for feature_name, detector_func in self.venue_features.items():
                task = asyncio.create_task(
                    self._run_detector(
                        detector_func, img_processed, feature_name)
                )
                detection_tasks.append(task)

            # 等待所有檢測完成
            results = await asyncio.gather(*detection_tasks)

            # 彙總分析結果
            return self._compile_results(results, img.shape)

        except Exception as e:
            logger.error(f"進階圖像分析錯誤: {str(e)}")
            return self._get_fallback_result()

    def _preprocess_image(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """圖像預處理"""
        processed = {}

        # 基礎處理
        processed['original'] = img.copy()
        processed['gray'] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed['hsv'] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 增強處理
        processed['enhanced'] = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        processed['denoised'] = cv2.bilateralFilter(img, 9, 75, 75)

        # 邊緣檢測
        processed['edges'] = cv2.Canny(processed['gray'], 50, 150)
        processed['edges_adaptive'] = cv2.adaptiveThreshold(
            processed['gray'], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return processed

    async def _run_detector(self, detector_func, img_data: Dict, feature_name: str) -> DetectionResult:
        """運行單個檢測器"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, detector_func, img_data
        )

    def _detect_olympic_rings(self, img_data: Dict) -> DetectionResult:
        """檢測奧運五環"""
        gray = img_data['gray']

        # 檢測圓形
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 30,
            param1=50, param2=30, minRadius=10, maxRadius=100
        )

        confidence = 0.0
        ring_count = 0

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            ring_count = len(circles)

            # 檢查是否有5個環形結構
            if ring_count >= 3:
                confidence = min(ring_count / 5.0 * 0.8, 0.8)

        return DetectionResult(
            feature_type="olympic_rings",
            confidence=confidence,
            description=f"檢測到 {ring_count} 個圓形結構"
        )

    def _detect_stadium_structure(self, img_data: Dict) -> DetectionResult:
        """檢測體育場結構"""
        edges = img_data['edges']

        # 檢測輪廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 分析大型結構
        large_structures = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # 調整閾值
                # 檢查形狀特徵
                approx = cv2.approxPolyDP(
                    contour, 0.02 * cv2.arcLength(contour, True), True)
                aspect_ratio = self._get_aspect_ratio(contour)

                large_structures.append({
                    'area': area,
                    'vertices': len(approx),
                    'aspect_ratio': aspect_ratio
                })

        # 計算信心度
        confidence = 0.0
        if large_structures:
            # 檢查是否有類似體育場的結構特徵
            stadium_like = sum(1 for s in large_structures
                               if s['area'] > 15000 and 0.5 <= s['aspect_ratio'] <= 2.0)
            confidence = min(stadium_like / len(large_structures) * 0.6, 0.6)

        return DetectionResult(
            feature_type="stadium_structure",
            confidence=confidence,
            description=f"檢測到 {len(large_structures)} 個大型結構"
        )

    def _detect_sports_equipment(self, img_data: Dict) -> DetectionResult:
        """檢測體育設備"""
        img = img_data['original']
        hsv = img_data['hsv']

        # 檢測特定形狀和顏色組合
        equipment_confidence = 0.0
        detected_items = []

        # 檢測白線 (運動場標線)
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        white_contours, _ = cv2.findContours(
            white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lines = [c for c in white_contours if cv2.contourArea(c) > 500]
        if lines:
            detected_items.append("運動場標線")
            equipment_confidence += 0.3

        # 檢測球門/籃框等矩形結構
        gray = img_data['gray']
        rectangles = cv2.findContours(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY +
                          cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )[0]

        goal_like = [r for r in rectangles
                     if 1000 < cv2.contourArea(r) < 10000
                     and 1.5 < self._get_aspect_ratio(r) < 4.0]

        if goal_like:
            detected_items.append("球門/籃框結構")
            equipment_confidence += 0.2

        return DetectionResult(
            feature_type="sports_equipment",
            confidence=min(equipment_confidence, 0.5),
            description=f"檢測到: {', '.join(detected_items) if detected_items else '無'}"
        )

    def _detect_text_signs(self, img_data: Dict) -> DetectionResult:
        """檢測文字標誌 (簡化版 OCR)"""
        gray = img_data['gray']

        # 使用形態學操作檢測文字區域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        text_regions = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

        # 檢測文字邊界框
        contours, _ = cv2.findContours(
            cv2.threshold(text_regions, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        text_like_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # 文字區域大小範圍
                aspect_ratio = self._get_aspect_ratio(contour)
                if 0.1 < aspect_ratio < 10.0:  # 文字的長寬比範圍
                    text_like_regions.append(contour)

        confidence = min(len(text_like_regions) / 20.0 * 0.4, 0.4)

        return DetectionResult(
            feature_type="venue_signage",
            confidence=confidence,
            description=f"檢測到 {len(text_like_regions)} 個文字區域"
        )

    def _detect_architecture(self, img_data: Dict) -> DetectionResult:
        """檢測建築特徵"""
        edges = img_data['edges']

        # 檢測直線 (建築物的特徵)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, threshold=80,
            minLineLength=100, maxLineGap=10
        )

        confidence = 0.0
        architectural_features = []

        if lines is not None:
            # 分析線條方向
            horizontal_lines = []
            vertical_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

                if abs(angle) < 15 or abs(angle) > 165:  # 水平線
                    horizontal_lines.append(line)
                elif 75 < abs(angle) < 105:  # 垂直線
                    vertical_lines.append(line)

            # 建築物通常有規律的水平和垂直線條
            if len(horizontal_lines) > 5 and len(vertical_lines) > 5:
                architectural_features.append("規律線條結構")
                confidence += 0.3

            # 檢測對稱性
            if self._check_symmetry(lines):
                architectural_features.append("對稱結構")
                confidence += 0.2

        return DetectionResult(
            feature_type="architectural_features",
            confidence=min(confidence, 0.5),
            description=f"建築特徵: {', '.join(architectural_features) if architectural_features else '無'}"
        )

    def _detect_crowd_patterns(self, img_data: Dict) -> DetectionResult:
        """檢測人群模式"""
        img = img_data['original']
        gray = img_data['gray']

        # 使用背景減法檢測移動物體 (簡化版)
        # 這裡使用色彩分布來推測人群

        # 檢測膚色範圍
        hsv = img_data['hsv']
        skin_mask = cv2.inRange(hsv, (0, 20, 70), (20, 255, 255))
        skin_area = cv2.countNonZero(skin_mask)

        # 檢測衣物顏色多樣性
        color_diversity = self._calculate_color_diversity(hsv)

        confidence = 0.0
        crowd_indicators = []

        # 如果有多樣的顏色分布，可能是人群
        if color_diversity > 0.3:
            crowd_indicators.append("顏色多樣性")
            confidence += 0.2

        # 如果有適量的膚色區域
        total_pixels = img.shape[0] * img.shape[1]
        skin_ratio = skin_area / total_pixels
        if 0.01 < skin_ratio < 0.1:
            crowd_indicators.append("膚色區域")
            confidence += 0.15

        return DetectionResult(
            feature_type="crowd_patterns",
            confidence=min(confidence, 0.35),
            description=f"人群指標: {', '.join(crowd_indicators) if crowd_indicators else '無'}"
        )

    def _get_aspect_ratio(self, contour) -> float:
        """計算輪廓的長寬比"""
        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if height == 0:
            return 0
        return max(width, height) / min(width, height)

    def _check_symmetry(self, lines) -> bool:
        """檢查線條的對稱性"""
        if lines is None or len(lines) < 4:
            return False

        # 簡化的對稱性檢查
        # 這裡可以實現更複雜的對稱性分析
        return len(lines) > 10  # 簡單判斷：多線條可能表示規律結構

    def _calculate_color_diversity(self, hsv_img) -> float:
        """計算色彩多樣性"""
        # 計算色調直方圖
        hist = cv2.calcHist([hsv_img], [0], None, [180], [0, 180])

        # 計算非零bin的數量
        non_zero_bins = np.count_nonzero(hist)

        # 歸一化到 0-1 範圍
        diversity = non_zero_bins / 180.0
        return diversity

    def _compile_results(self, results: List[DetectionResult], img_shape: Tuple) -> Dict:
        """編譯最終結果"""
        total_confidence = 0.0
        detected_objects = []
        detailed_analysis = {}

        for result in results:
            if result.confidence > 0.1:  # 只考慮有意義的檢測
                total_confidence += result.confidence
                detected_objects.append(result.feature_type)
                detailed_analysis[result.feature_type] = {
                    'confidence': result.confidence,
                    'description': result.description
                }

        # 計算最終信心度
        final_confidence = min(total_confidence, 1.0)

        # 判斷是否檢測到場館
        is_venue_detected = final_confidence > 0.5

        # 風險等級評估
        if final_confidence > 0.8:
            risk_level = "high"
        elif final_confidence > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            'is_venue_detected': is_venue_detected,
            'confidence': final_confidence,
            'detected_objects': detected_objects,
            'risk_level': risk_level,
            'detailed_analysis': detailed_analysis,
            'image_size': f"{img_shape[1]}x{img_shape[0]}"
        }

    def _get_fallback_result(self) -> Dict:
        """回退結果"""
        return {
            'is_venue_detected': False,
            'confidence': 0.0,
            'detected_objects': [],
            'risk_level': 'low',
            'detailed_analysis': {},
            'error': 'Analysis failed'
        }

# 使用示例


async def analyze_with_advanced_ai(image_data: bytes) -> Dict:
    """使用進階 AI 分析圖像"""
    detector = AdvancedVenueDetector()
    return await detector.analyze_image_advanced(image_data)
