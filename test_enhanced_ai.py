#!/usr/bin/env python3
"""
Olympic CamGuard - 增強 AI 測試腳本
展示最新的圖像分析能力
"""

import asyncio
import time
import cv2
import numpy as np
from pathlib import Path
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_images():
    """創建測試圖像"""
    test_images = {}

    # 1. 奧運五環測試圖像
    olympic_rings_img = np.zeros((400, 600, 3), dtype=np.uint8)
    colors = [(255, 0, 0), (255, 255, 0), (0, 0, 0),
              (0, 255, 0), (0, 0, 255)]  # BGR format
    centers = [(120, 150), (200, 150), (280, 150), (160, 220), (240, 220)]

    for i, (center, color) in enumerate(zip(centers, colors)):
        cv2.circle(olympic_rings_img, center, 50, color, 8)

    _, encoded_rings = cv2.imencode('.jpg', olympic_rings_img)
    test_images['olympic_rings'] = encoded_rings.tobytes()

    # 2. 體育場建築測試圖像
    stadium_img = np.zeros((400, 600, 3), dtype=np.uint8)
    # 繪製橢圓體育場
    cv2.ellipse(stadium_img, (300, 200), (250, 150),
                0, 0, 360, (128, 128, 128), 5)
    # 繪製看台階梯
    for i in range(10):
        y = 100 + i * 15
        cv2.line(stadium_img, (50, y), (550, y), (200, 200, 200), 2)

    _, encoded_stadium = cv2.imencode('.jpg', stadium_img)
    test_images['stadium'] = encoded_stadium.tobytes()

    # 3. 運動場標記測試圖像
    field_img = np.zeros((400, 600, 3), dtype=np.uint8)
    # 綠色背景
    field_img[:] = (0, 128, 0)
    # 白色標線
    cv2.rectangle(field_img, (50, 50), (550, 350), (255, 255, 255), 3)
    cv2.circle(field_img, (300, 200), 80, (255, 255, 255), 3)
    cv2.line(field_img, (300, 50), (300, 350), (255, 255, 255), 3)

    _, encoded_field = cv2.imencode('.jpg', field_img)
    test_images['sports_field'] = encoded_field.tobytes()

    # 4. 混合場景測試圖像
    mixed_img = np.zeros((400, 600, 3), dtype=np.uint8)
    # 體育場結構
    cv2.ellipse(mixed_img, (300, 200), (280, 180),
                0, 0, 360, (100, 100, 100), 3)
    # 一些圓形
    for i in range(3):
        cv2.circle(mixed_img, (150 + i*100, 100), 30, (0, 255, 255), 5)
    # 一些矩形結構
    cv2.rectangle(mixed_img, (100, 300), (500, 380), (255, 255, 255), 2)

    _, encoded_mixed = cv2.imencode('.jpg', mixed_img)
    test_images['mixed_scene'] = encoded_mixed.tobytes()

    return test_images


async def test_enhanced_ai_system():
    """測試增強AI系統"""
    try:
        # 導入增強AI模組
        from enhanced_ai_system import analyze_with_ultra_ai

        print("🚀 開始測試超先進 AI 檢測系統...")
        print("=" * 60)

        # 創建測試圖像
        test_images = create_test_images()

        # 測試每個圖像
        for image_name, image_data in test_images.items():
            print(f"\n📸 測試圖像: {image_name}")
            print("-" * 40)

            start_time = time.time()

            # 使用超先進AI分析
            result = await analyze_with_ultra_ai(image_data)

            analysis_time = time.time() - start_time

            # 顯示結果
            print(f"✅ 場館檢測: {'是' if result['is_venue_detected'] else '否'}")
            print(f"🎯 信心度: {result['confidence']:.3f}")
            print(f"⚠️  風險等級: {result['risk_level']}")
            print(f"🔍 檢測對象: {', '.join(result['detected_objects'])}")
            print(f"⏱️  處理時間: {analysis_time:.3f}秒")

            if 'detailed_analysis' in result:
                detailed = result['detailed_analysis']
                print(f"📊 詳細分析:")

                if 'feature_scores' in detailed:
                    print("   特徵分數:")
                    for feature, score in detailed['feature_scores'].items():
                        print(f"     • {feature}: {score:.3f}")

                if 'fusion_layers' in detailed:
                    fusion = detailed['fusion_layers']
                    print(f"   融合結果: 最終={fusion.get('final', 0):.3f}, "
                          f"調整後={fusion.get('adjusted', 0):.3f}")

                if 'image_quality_factor' in detailed:
                    print(f"   圖像品質因子: {detailed['image_quality_factor']:.3f}")

        print("\n" + "=" * 60)
        print("✅ 測試完成！")

    except ImportError as e:
        print(f"❌ 無法導入增強AI模組: {e}")
        print("💡 使用基礎AI進行測試...")
        await test_basic_ai_system()
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")


async def test_basic_ai_system():
    """測試基礎AI系統"""
    try:
        # 導入主系統
        from main import analyze_image_for_venue, AIAnalysisResult

        print("🔧 使用基礎 AI 檢測系統...")

        # 創建測試圖像
        test_images = create_test_images()

        for image_name, image_data in test_images.items():
            print(f"\n📸 測試圖像: {image_name}")
            print("-" * 40)

            start_time = time.time()

            # 使用基礎AI分析
            result = analyze_image_for_venue(image_data)

            analysis_time = time.time() - start_time

            # 顯示結果
            print(f"✅ 場館檢測: {'是' if result.is_venue_detected else '否'}")
            print(f"🎯 信心度: {result.confidence:.3f}")
            print(f"⚠️  風險等級: {result.risk_level}")
            print(f"🔍 檢測對象: {', '.join(result.detected_objects)}")
            print(f"⏱️  處理時間: {analysis_time:.3f}秒")

            if result.detailed_analysis:
                detailed = result.detailed_analysis
                print(f"📊 詳細分析:")

                if 'feature_scores' in detailed:
                    print("   特徵分數:")
                    for feature, score in detailed['feature_scores'].items():
                        print(f"     • {feature}: {score:.3f}")

        print("\n" + "=" * 40)
        print("✅ 基礎測試完成！")

    except Exception as e:
        print(f"❌ 基礎測試失敗: {e}")


def performance_comparison():
    """性能比較測試"""
    print("\n🏁 性能比較測試")
    print("=" * 50)

    test_images = create_test_images()
    sample_image = test_images['mixed_scene']

    # 測試次數
    test_rounds = 5

    try:
        # 測試增強AI
        print("🚀 測試超先進AI...")
        from enhanced_ai_system import analyze_with_ultra_ai_sync

        enhanced_times = []
        for i in range(test_rounds):
            start_time = time.time()
            result = analyze_with_ultra_ai_sync(sample_image)
            enhanced_times.append(time.time() - start_time)
            print(f"   第{i+1}輪: {enhanced_times[-1]:.3f}秒")

        avg_enhanced = sum(enhanced_times) / len(enhanced_times)
        print(f"📊 超先進AI平均時間: {avg_enhanced:.3f}秒")

    except ImportError:
        print("⚠️  超先進AI模組不可用")
        avg_enhanced = 0

    try:
        # 測試基礎AI
        print("\n🔧 測試基礎AI...")
        from main import analyze_image_for_venue

        basic_times = []
        for i in range(test_rounds):
            start_time = time.time()
            result = analyze_image_for_venue(sample_image)
            basic_times.append(time.time() - start_time)
            print(f"   第{i+1}輪: {basic_times[-1]:.3f}秒")

        avg_basic = sum(basic_times) / len(basic_times)
        print(f"📊 基礎AI平均時間: {avg_basic:.3f}秒")

        if avg_enhanced > 0:
            speedup = avg_basic / avg_enhanced
            print(
                f"🏃 性能比較: 超先進AI相對於基礎AI {'快' if speedup > 1 else '慢'} {abs(speedup-1)*100:.1f}%")

    except Exception as e:
        print(f"❌ 基礎AI測試失敗: {e}")


def accuracy_test():
    """準確度測試"""
    print("\n🎯 準確度測試")
    print("=" * 50)

    # 預期結果
    expected_results = {
        'olympic_rings': {'venue_detected': True, 'min_confidence': 0.5},
        'stadium': {'venue_detected': True, 'min_confidence': 0.4},
        'sports_field': {'venue_detected': True, 'min_confidence': 0.4},
        'mixed_scene': {'venue_detected': True, 'min_confidence': 0.3}
    }

    test_images = create_test_images()

    try:
        from enhanced_ai_system import analyze_with_ultra_ai_sync

        print("🧪 測試超先進AI準確度...")
        correct_predictions = 0
        total_tests = len(expected_results)

        for image_name, image_data in test_images.items():
            if image_name in expected_results:
                result = analyze_with_ultra_ai_sync(image_data)
                expected = expected_results[image_name]

                venue_correct = result['is_venue_detected'] == expected['venue_detected']
                confidence_ok = result['confidence'] >= expected['min_confidence']

                if venue_correct and confidence_ok:
                    correct_predictions += 1
                    status = "✅"
                else:
                    status = "❌"

                print(f"   {status} {image_name}: 檢測={result['is_venue_detected']}, "
                      f"信心度={result['confidence']:.3f}")

        accuracy = correct_predictions / total_tests * 100
        print(
            f"📊 超先進AI準確度: {accuracy:.1f}% ({correct_predictions}/{total_tests})")

    except ImportError:
        print("⚠️  超先進AI不可用，跳過準確度測試")


async def main():
    """主函數"""
    print("🏅 Olympic CamGuard - AI 檢測系統測試")
    print("=" * 60)

    # 主要測試
    await test_enhanced_ai_system()

    # 性能測試
    performance_comparison()

    # 準確度測試
    accuracy_test()

    print("\n🎉 所有測試完成！")

if __name__ == "__main__":
    # 運行異步測試
    asyncio.run(main())
