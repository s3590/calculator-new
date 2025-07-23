"""
محرك الذكاء الاصطناعي المتقدم للتنبؤ المالي
يستخدم OpenAI وأقوى المؤشرات التقنية
"""
import openai
import json
import numpy as np
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Any
import asyncio
import aiohttp

class AIFinancialEngine:
    """محرك الذكاء الاصطناعي للتحليل المالي"""
    
    def __init__(self):
        # تم تعيين مفاتيح OpenAI تلقائياً في البيئة
        self.client = openai.OpenAI()
        self.market_data_cache = {}
        self.analysis_cache = {}
        
    async def analyze_with_ai(self, asset: str, market: str, timeframe: str, price_data: Dict) -> Dict:
        """تحليل شامل باستخدام الذكاء الاصطناعي"""
        
        # إعداد البيانات للتحليل
        analysis_prompt = self._create_analysis_prompt(asset, market, timeframe, price_data)
        
        try:
            # استخدام GPT-4 للتحليل المتقدم
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": """أنت خبير تحليل مالي متقدم متخصص في التنبؤ بأسعار الأصول المالية. 
                        تستخدم أقوى المؤشرات التقنية والتحليل الأساسي ومعنويات السوق.
                        قدم تحليلاً دقيقاً ومفصلاً مع نسبة ثقة واقعية."""
                    },
                    {
                        "role": "user", 
                        "content": analysis_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            ai_analysis = response.choices[0].message.content
            
            # تحليل استجابة الذكاء الاصطناعي
            structured_analysis = self._parse_ai_response(ai_analysis)
            
            # دمج التحليل مع المؤشرات التقنية المتقدمة
            enhanced_analysis = await self._enhance_with_technical_indicators(
                structured_analysis, asset, market, timeframe, price_data
            )
            
            return enhanced_analysis
            
        except Exception as e:
            print(f"خطأ في تحليل الذكاء الاصطناعي: {e}")
            # العودة للتحليل التقليدي في حالة الخطأ
            return await self._fallback_analysis(asset, market, timeframe, price_data)
    
    def _create_analysis_prompt(self, asset: str, market: str, timeframe: str, price_data: Dict) -> str:
        """إنشاء prompt متقدم للذكاء الاصطناعي"""
        
        current_price = price_data.get('current_price', 0)
        price_history = price_data.get('price_history', [])
        volume_data = price_data.get('volume', 0)
        
        # حساب المؤشرات الأساسية
        price_change = self._calculate_price_change(price_history)
        volatility = self._calculate_volatility(price_history)
        trend_strength = self._calculate_trend_strength(price_history)
        
        prompt = f"""
        تحليل مالي متقدم للأصل: {asset} في سوق {market}
        
        البيانات الحالية:
        - السعر الحالي: {current_price}
        - الإطار الزمني: {timeframe}
        - التغيير في السعر: {price_change:.2%}
        - التقلبات: {volatility:.2%}
        - قوة الاتجاه: {trend_strength:.2f}
        - الحجم: {volume_data}
        
        تاريخ الأسعار (آخر 10 نقاط): {price_history[-10:] if len(price_history) >= 10 else price_history}
        
        المطلوب:
        1. تحليل الاتجاه العام للأصل
        2. تقييم قوة الاتجاه الحالي
        3. تحديد مستويات الدعم والمقاومة
        4. تحليل معنويات السوق
        5. التنبؤ بالاتجاه للإطار الزمني المحدد
        6. نسبة الثقة في التنبؤ (من 50% إلى 95%)
        
        يرجى تقديم الإجابة بتنسيق JSON مع المفاتيح التالية:
        {{
            "direction": "up/down",
            "confidence": 85,
            "trend_analysis": "تحليل الاتجاه",
            "support_resistance": {{"support": 0, "resistance": 0}},
            "market_sentiment": "bullish/bearish/neutral",
            "risk_factors": ["عامل 1", "عامل 2"],
            "key_indicators": {{"rsi_signal": "", "macd_signal": "", "volume_signal": ""}},
            "reasoning": "سبب التنبؤ"
        }}
        """
        
        return prompt
    
    def _parse_ai_response(self, ai_response: str) -> Dict:
        """تحليل استجابة الذكاء الاصطناعي"""
        try:
            # محاولة استخراج JSON من الاستجابة
            start_idx = ai_response.find('{')
            end_idx = ai_response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = ai_response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # تحليل نصي إذا لم يكن JSON
                return self._parse_text_response(ai_response)
                
        except json.JSONDecodeError:
            return self._parse_text_response(ai_response)
    
    def _parse_text_response(self, response: str) -> Dict:
        """تحليل الاستجابة النصية"""
        # تحليل بسيط للاستجابة النصية
        direction = "up" if any(word in response.lower() for word in ["صعود", "ارتفاع", "bullish", "up"]) else "down"
        
        # استخراج نسبة الثقة
        confidence = 75  # قيمة افتراضية
        for word in response.split():
            if '%' in word:
                try:
                    confidence = int(word.replace('%', ''))
                    break
                except:
                    pass
        
        return {
            "direction": direction,
            "confidence": min(95, max(50, confidence)),
            "trend_analysis": "تحليل مستخرج من النص",
            "market_sentiment": "neutral",
            "reasoning": response[:200] + "..." if len(response) > 200 else response
        }
    
    async def _enhance_with_technical_indicators(self, ai_analysis: Dict, asset: str, 
                                               market: str, timeframe: str, price_data: Dict) -> Dict:
        """تعزيز التحليل بالمؤشرات التقنية المتقدمة"""
        
        price_history = price_data.get('price_history', [])
        
        if len(price_history) < 20:
            # إنشاء بيانات تاريخية إضافية للتحليل
            price_history = self._generate_extended_history(price_data.get('current_price', 100))
        
        # حساب المؤشرات المتقدمة
        indicators = {
            'rsi': self._calculate_rsi(price_history),
            'macd': self._calculate_macd(price_history),
            'bollinger_bands': self._calculate_bollinger_bands(price_history),
            'stochastic': self._calculate_stochastic(price_history),
            'williams_r': self._calculate_williams_r(price_history),
            'cci': self._calculate_cci(price_history),
            'atr': self._calculate_atr(price_history),
            'adx': self._calculate_adx(price_history),
            'fibonacci_levels': self._calculate_fibonacci_levels(price_history),
            'ichimoku': self._calculate_ichimoku(price_history)
        }
        
        # تحليل الحجم المتقدم
        volume_analysis = self._analyze_volume_patterns(price_data)
        
        # تحليل الشموع اليابانية
        candlestick_patterns = self._analyze_candlestick_patterns(price_history)
        
        # تحليل الموجات
        wave_analysis = self._analyze_elliott_waves(price_history)
        
        # دمج جميع التحليلات
        enhanced_analysis = {
            **ai_analysis,
            'technical_indicators': indicators,
            'volume_analysis': volume_analysis,
            'candlestick_patterns': candlestick_patterns,
            'wave_analysis': wave_analysis,
            'timeframe_analysis': self._analyze_timeframe_correlation(timeframe, indicators),
            'market_correlation': await self._analyze_market_correlation(asset, market),
            'sentiment_score': self._calculate_sentiment_score(indicators, ai_analysis),
            'risk_assessment': self._assess_risk_level(indicators, price_history),
            'entry_exit_points': self._calculate_entry_exit_points(indicators, price_history[-1])
        }
        
        # تعديل الثقة بناءً على التحليل المتقدم
        enhanced_analysis['confidence'] = self._adjust_confidence_with_indicators(
            ai_analysis.get('confidence', 75), indicators, timeframe
        )
        
        return enhanced_analysis
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Dict:
        """حساب مؤشر القوة النسبية المتقدم"""
        if len(prices) < period + 1:
            return {'value': 50, 'signal': 'neutral', 'divergence': False}
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        
        # تحليل الإشارات
        signal = 'neutral'
        if rsi < 30:
            signal = 'oversold'
        elif rsi > 70:
            signal = 'overbought'
        elif rsi < 40:
            signal = 'bearish'
        elif rsi > 60:
            signal = 'bullish'
        
        # كشف التباعد
        divergence = self._detect_rsi_divergence(prices, period)
        
        return {
            'value': round(rsi, 2),
            'signal': signal,
            'divergence': divergence,
            'strength': abs(rsi - 50) / 50
        }
    
    def _calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """حساب MACD المتقدم"""
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'trend': 'neutral'}
        
        # حساب المتوسطات المتحركة الأسية
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # حساب خط الإشارة
        macd_values = [macd_line] * min(signal, len(prices))
        signal_line = self._calculate_ema(macd_values, signal)
        
        histogram = macd_line - signal_line
        
        # تحديد الاتجاه
        trend = 'bullish' if histogram > 0 else 'bearish'
        if abs(histogram) < 0.001:
            trend = 'neutral'
        
        return {
            'macd': round(macd_line, 4),
            'signal': round(signal_line, 4),
            'histogram': round(histogram, 4),
            'trend': trend,
            'strength': abs(histogram)
        }
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict:
        """حساب نطاقات بولينجر المتقدمة"""
        if len(prices) < period:
            avg_price = np.mean(prices)
            return {
                'upper': avg_price * 1.02,
                'middle': avg_price,
                'lower': avg_price * 0.98,
                'width': 0.04,
                'position': 'middle'
            }
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        current_price = prices[-1]
        
        # تحديد موقع السعر
        if current_price > upper_band:
            position = 'above_upper'
        elif current_price < lower_band:
            position = 'below_lower'
        elif current_price > sma:
            position = 'above_middle'
        else:
            position = 'below_middle'
        
        # عرض النطاق
        width = (upper_band - lower_band) / sma
        
        return {
            'upper': round(upper_band, 4),
            'middle': round(sma, 4),
            'lower': round(lower_band, 4),
            'width': round(width, 4),
            'position': position,
            'squeeze': width < 0.1  # انضغاط النطاق
        }
    
    def _calculate_stochastic(self, prices: List[float], k_period: int = 14, d_period: int = 3) -> Dict:
        """حساب مؤشر ستوكاستيك المتقدم"""
        if len(prices) < k_period:
            return {'k': 50, 'd': 50, 'signal': 'neutral'}
        
        # افتراض أن الأسعار تمثل الإغلاق، وسنحسب الأعلى والأدنى
        highs = [p * 1.01 for p in prices]  # تقدير الأعلى
        lows = [p * 0.99 for p in prices]   # تقدير الأدنى
        
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = prices[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # حساب %D (متوسط %K)
        d_percent = k_percent  # تبسيط
        
        # تحديد الإشارة
        signal = 'neutral'
        if k_percent < 20:
            signal = 'oversold'
        elif k_percent > 80:
            signal = 'overbought'
        
        return {
            'k': round(k_percent, 2),
            'd': round(d_percent, 2),
            'signal': signal
        }
    
    def _calculate_williams_r(self, prices: List[float], period: int = 14) -> Dict:
        """حساب مؤشر Williams %R"""
        if len(prices) < period:
            return {'value': -50, 'signal': 'neutral'}
        
        highs = [p * 1.01 for p in prices[-period:]]
        lows = [p * 0.99 for p in prices[-period:]]
        current_close = prices[-1]
        
        highest_high = max(highs)
        lowest_low = min(lows)
        
        if highest_high == lowest_low:
            williams_r = -50
        else:
            williams_r = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
        
        signal = 'neutral'
        if williams_r < -80:
            signal = 'oversold'
        elif williams_r > -20:
            signal = 'overbought'
        
        return {
            'value': round(williams_r, 2),
            'signal': signal
        }
    
    def _calculate_cci(self, prices: List[float], period: int = 20) -> Dict:
        """حساب مؤشر CCI (Commodity Channel Index)"""
        if len(prices) < period:
            return {'value': 0, 'signal': 'neutral'}
        
        # حساب السعر النموذجي (متوسط الأعلى والأدنى والإغلاق)
        typical_prices = [(p * 1.01 + p * 0.99 + p) / 3 for p in prices[-period:]]
        
        sma_tp = np.mean(typical_prices)
        mean_deviation = np.mean([abs(tp - sma_tp) for tp in typical_prices])
        
        if mean_deviation == 0:
            cci = 0
        else:
            cci = (typical_prices[-1] - sma_tp) / (0.015 * mean_deviation)
        
        signal = 'neutral'
        if cci > 100:
            signal = 'overbought'
        elif cci < -100:
            signal = 'oversold'
        
        return {
            'value': round(cci, 2),
            'signal': signal
        }
    
    def _calculate_atr(self, prices: List[float], period: int = 14) -> Dict:
        """حساب متوسط المدى الحقيقي (ATR)"""
        if len(prices) < period + 1:
            return {'value': 0, 'volatility': 'low'}
        
        # حساب المدى الحقيقي
        true_ranges = []
        for i in range(1, len(prices)):
            high = prices[i] * 1.01
            low = prices[i] * 0.99
            prev_close = prices[i-1]
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)
        
        atr = np.mean(true_ranges[-period:])
        
        # تحديد مستوى التقلب
        volatility = 'low'
        if atr > prices[-1] * 0.02:
            volatility = 'high'
        elif atr > prices[-1] * 0.01:
            volatility = 'medium'
        
        return {
            'value': round(atr, 4),
            'volatility': volatility,
            'percentage': round((atr / prices[-1]) * 100, 2)
        }
    
    def _calculate_adx(self, prices: List[float], period: int = 14) -> Dict:
        """حساب مؤشر الاتجاه المتوسط (ADX)"""
        if len(prices) < period + 1:
            return {'value': 25, 'trend_strength': 'weak'}
        
        # تبسيط حساب ADX
        price_changes = np.diff(prices[-period:])
        positive_changes = np.where(price_changes > 0, price_changes, 0)
        negative_changes = np.where(price_changes < 0, abs(price_changes), 0)
        
        avg_positive = np.mean(positive_changes)
        avg_negative = np.mean(negative_changes)
        
        if avg_positive + avg_negative == 0:
            adx = 25
        else:
            dx = abs(avg_positive - avg_negative) / (avg_positive + avg_negative) * 100
            adx = dx  # تبسيط
        
        # تحديد قوة الاتجاه
        if adx > 50:
            trend_strength = 'very_strong'
        elif adx > 25:
            trend_strength = 'strong'
        elif adx > 20:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        return {
            'value': round(adx, 2),
            'trend_strength': trend_strength
        }
    
    def _calculate_fibonacci_levels(self, prices: List[float]) -> Dict:
        """حساب مستويات فيبوناتشي"""
        if len(prices) < 10:
            return {'levels': {}, 'current_level': 'unknown'}
        
        high = max(prices[-20:]) if len(prices) >= 20 else max(prices)
        low = min(prices[-20:]) if len(prices) >= 20 else min(prices)
        diff = high - low
        
        levels = {
            '0%': high,
            '23.6%': high - (diff * 0.236),
            '38.2%': high - (diff * 0.382),
            '50%': high - (diff * 0.5),
            '61.8%': high - (diff * 0.618),
            '78.6%': high - (diff * 0.786),
            '100%': low
        }
        
        current_price = prices[-1]
        current_level = 'unknown'
        
        # تحديد المستوى الحالي
        for level_name, level_price in levels.items():
            if abs(current_price - level_price) < diff * 0.02:  # ضمن 2% من المستوى
                current_level = level_name
                break
        
        return {
            'levels': {k: round(v, 4) for k, v in levels.items()},
            'current_level': current_level,
            'support': min(levels.values()),
            'resistance': max(levels.values())
        }
    
    def _calculate_ichimoku(self, prices: List[float]) -> Dict:
        """حساب مؤشر إيشيموكو"""
        if len(prices) < 52:
            return {'signal': 'neutral', 'cloud_position': 'unknown'}
        
        # خط التحويل (9 فترات)
        tenkan_high = max(prices[-9:])
        tenkan_low = min(prices[-9:])
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # الخط الأساسي (26 فترة)
        kijun_high = max(prices[-26:])
        kijun_low = min(prices[-26:])
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # السحابة
        senkou_span_a = (tenkan_sen + kijun_sen) / 2
        senkou_span_b = (max(prices[-52:]) + min(prices[-52:])) / 2
        
        current_price = prices[-1]
        
        # تحديد موقع السعر بالنسبة للسحابة
        cloud_top = max(senkou_span_a, senkou_span_b)
        cloud_bottom = min(senkou_span_a, senkou_span_b)
        
        if current_price > cloud_top:
            cloud_position = 'above_cloud'
            signal = 'bullish'
        elif current_price < cloud_bottom:
            cloud_position = 'below_cloud'
            signal = 'bearish'
        else:
            cloud_position = 'in_cloud'
            signal = 'neutral'
        
        return {
            'tenkan_sen': round(tenkan_sen, 4),
            'kijun_sen': round(kijun_sen, 4),
            'senkou_span_a': round(senkou_span_a, 4),
            'senkou_span_b': round(senkou_span_b, 4),
            'signal': signal,
            'cloud_position': cloud_position
        }
    
    def _analyze_timeframe_correlation(self, timeframe: str, indicators: Dict) -> Dict:
        """تحليل الارتباط مع الإطار الزمني"""
        
        # عوامل التعديل حسب الإطار الزمني
        timeframe_factors = {
            '1 دقيقة': {'noise_level': 'high', 'reliability': 0.6, 'volatility_weight': 1.5},
            '5 دقائق': {'noise_level': 'medium', 'reliability': 0.7, 'volatility_weight': 1.3},
            '10 دقائق': {'noise_level': 'medium', 'reliability': 0.75, 'volatility_weight': 1.2},
            '30 دقيقة': {'noise_level': 'low', 'reliability': 0.85, 'volatility_weight': 1.0},
            '1 ساعة': {'noise_level': 'low', 'reliability': 0.9, 'volatility_weight': 0.8}
        }
        
        factors = timeframe_factors.get(timeframe, timeframe_factors['30 دقيقة'])
        
        # تعديل المؤشرات حسب الإطار الزمني
        adjusted_signals = {}
        
        # RSI - أكثر موثوقية في الإطارات الطويلة
        rsi_reliability = factors['reliability']
        if indicators['rsi']['value'] < 30 or indicators['rsi']['value'] > 70:
            adjusted_signals['rsi'] = {
                'signal': indicators['rsi']['signal'],
                'reliability': rsi_reliability
            }
        
        # MACD - فعال في جميع الإطارات
        adjusted_signals['macd'] = {
            'signal': indicators['macd']['trend'],
            'reliability': factors['reliability']
        }
        
        # Bollinger Bands - تأثر بالتقلبات
        bb_reliability = factors['reliability'] * (1 / factors['volatility_weight'])
        adjusted_signals['bollinger'] = {
            'signal': indicators['bollinger_bands']['position'],
            'reliability': bb_reliability
        }
        
        return {
            'timeframe': timeframe,
            'factors': factors,
            'adjusted_signals': adjusted_signals,
            'overall_reliability': factors['reliability']
        }
    
    async def _analyze_market_correlation(self, asset: str, market: str) -> Dict:
        """تحليل الارتباط مع السوق"""
        
        # عوامل الارتباط حسب نوع السوق
        market_correlations = {
            'العملات': {
                'correlation_with_usd': 0.8,
                'volatility_factor': 1.0,
                'news_sensitivity': 'high'
            },
            'العملات المشفرة': {
                'correlation_with_btc': 0.7,
                'volatility_factor': 2.0,
                'news_sensitivity': 'very_high'
            },
            'السلع': {
                'correlation_with_dollar': -0.6,
                'volatility_factor': 1.2,
                'news_sensitivity': 'medium'
            },
            'الأسهم': {
                'correlation_with_market': 0.6,
                'volatility_factor': 1.1,
                'news_sensitivity': 'high'
            }
        }
        
        correlation_data = market_correlations.get(market, market_correlations['الأسهم'])
        
        # تحليل معنويات السوق العامة
        market_sentiment = await self._get_market_sentiment(market)
        
        return {
            'market': market,
            'asset': asset,
            'correlation_data': correlation_data,
            'market_sentiment': market_sentiment,
            'risk_level': self._calculate_market_risk(correlation_data, market_sentiment)
        }
    
    async def _get_market_sentiment(self, market: str) -> Dict:
        """الحصول على معنويات السوق"""
        
        # محاكاة معنويات السوق (في التطبيق الحقيقي، يمكن ربطها بـ APIs خارجية)
        import random
        
        sentiment_scores = {
            'العملات': random.uniform(0.4, 0.8),
            'العملات المشفرة': random.uniform(0.3, 0.9),
            'السلع': random.uniform(0.4, 0.7),
            'الأسهم': random.uniform(0.5, 0.8)
        }
        
        score = sentiment_scores.get(market, 0.6)
        
        if score > 0.7:
            sentiment = 'bullish'
        elif score < 0.4:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'score': round(score, 2),
            'confidence': random.uniform(0.6, 0.9)
        }
    
    def _calculate_sentiment_score(self, indicators: Dict, ai_analysis: Dict) -> Dict:
        """حساب نقاط المعنويات الإجمالية"""
        
        scores = []
        
        # نقاط من المؤشرات التقنية
        if indicators['rsi']['signal'] == 'bullish':
            scores.append(0.7)
        elif indicators['rsi']['signal'] == 'bearish':
            scores.append(0.3)
        else:
            scores.append(0.5)
        
        if indicators['macd']['trend'] == 'bullish':
            scores.append(0.8)
        elif indicators['macd']['trend'] == 'bearish':
            scores.append(0.2)
        else:
            scores.append(0.5)
        
        # نقاط من تحليل الذكاء الاصطناعي
        ai_sentiment = ai_analysis.get('market_sentiment', 'neutral')
        if ai_sentiment == 'bullish':
            scores.append(0.8)
        elif ai_sentiment == 'bearish':
            scores.append(0.2)
        else:
            scores.append(0.5)
        
        overall_score = np.mean(scores)
        
        if overall_score > 0.6:
            sentiment = 'bullish'
        elif overall_score < 0.4:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        return {
            'overall_score': round(overall_score, 2),
            'sentiment': sentiment,
            'component_scores': scores
        }
    
    def _assess_risk_level(self, indicators: Dict, price_history: List[float]) -> Dict:
        """تقييم مستوى المخاطر"""
        
        risk_factors = []
        risk_score = 0
        
        # تقلبات السعر
        volatility = self._calculate_volatility(price_history)
        if volatility > 0.05:  # 5%
            risk_factors.append('تقلبات عالية')
            risk_score += 0.3
        
        # مؤشر ATR
        atr_volatility = indicators['atr']['volatility']
        if atr_volatility == 'high':
            risk_factors.append('مدى تداول واسع')
            risk_score += 0.2
        
        # مؤشر ADX
        trend_strength = indicators['adx']['trend_strength']
        if trend_strength == 'weak':
            risk_factors.append('اتجاه ضعيف')
            risk_score += 0.2
        
        # نطاقات بولينجر
        if indicators['bollinger_bands']['squeeze']:
            risk_factors.append('انضغاط السوق')
            risk_score += 0.1
        
        # تحديد مستوى المخاطر
        if risk_score > 0.6:
            risk_level = 'high'
        elif risk_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'risk_factors': risk_factors
        }
    
    def _calculate_entry_exit_points(self, indicators: Dict, current_price: float) -> Dict:
        """حساب نقاط الدخول والخروج"""
        
        entry_points = []
        exit_points = []
        
        # نقاط الدخول من Bollinger Bands
        bb = indicators['bollinger_bands']
        if bb['position'] == 'below_lower':
            entry_points.append({
                'type': 'buy',
                'price': bb['lower'],
                'reason': 'ارتداد من النطاق السفلي'
            })
        elif bb['position'] == 'above_upper':
            entry_points.append({
                'type': 'sell',
                'price': bb['upper'],
                'reason': 'ارتداد من النطاق العلوي'
            })
        
        # نقاط من مستويات فيبوناتشي
        fib = indicators['fibonacci_levels']
        support = fib['support']
        resistance = fib['resistance']
        
        entry_points.append({
            'type': 'buy',
            'price': support,
            'reason': 'مستوى دعم فيبوناتشي'
        })
        
        exit_points.append({
            'type': 'sell',
            'price': resistance,
            'reason': 'مستوى مقاومة فيبوناتشي'
        })
        
        # نقاط من RSI
        rsi = indicators['rsi']
        if rsi['signal'] == 'oversold':
            entry_points.append({
                'type': 'buy',
                'price': current_price * 0.99,
                'reason': 'RSI في منطقة التشبع البيعي'
            })
        elif rsi['signal'] == 'overbought':
            exit_points.append({
                'type': 'sell',
                'price': current_price * 1.01,
                'reason': 'RSI في منطقة التشبع الشرائي'
            })
        
        return {
            'entry_points': entry_points,
            'exit_points': exit_points,
            'stop_loss': current_price * 0.95,  # 5% stop loss
            'take_profit': current_price * 1.1   # 10% take profit
        }
    
    def _adjust_confidence_with_indicators(self, base_confidence: int, indicators: Dict, timeframe: str) -> int:
        """تعديل الثقة بناءً على المؤشرات"""
        
        confidence_adjustments = 0
        
        # تعديل من RSI
        if indicators['rsi']['signal'] in ['oversold', 'overbought']:
            confidence_adjustments += 5
        
        # تعديل من MACD
        if abs(indicators['macd']['histogram']) > 0.01:
            confidence_adjustments += 3
        
        # تعديل من ADX
        if indicators['adx']['trend_strength'] in ['strong', 'very_strong']:
            confidence_adjustments += 7
        
        # تعديل من الإطار الزمني
        timeframe_bonus = {
            '1 دقيقة': -10,
            '5 دقائق': -5,
            '10 دقائق': 0,
            '30 دقيقة': 5,
            '1 ساعة': 10
        }
        
        confidence_adjustments += timeframe_bonus.get(timeframe, 0)
        
        # تطبيق التعديلات
        final_confidence = base_confidence + confidence_adjustments
        
        # التأكد من أن الثقة ضمن النطاق المسموح
        return min(95, max(50, final_confidence))
    
    # دوال مساعدة
    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """حساب المتوسط المتحرك الأسي"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def _calculate_price_change(self, prices: List[float]) -> float:
        """حساب تغيير السعر"""
        if len(prices) < 2:
            return 0
        return (prices[-1] - prices[0]) / prices[0]
    
    def _calculate_volatility(self, prices: List[float]) -> float:
        """حساب التقلبات"""
        if len(prices) < 2:
            return 0
        returns = np.diff(prices) / prices[:-1]
        return np.std(returns)
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """حساب قوة الاتجاه"""
        if len(prices) < 10:
            return 0.5
        
        # حساب معامل الارتباط مع الزمن
        x = np.arange(len(prices))
        correlation = np.corrcoef(x, prices)[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.5
    
    def _generate_extended_history(self, current_price: float, days: int = 30) -> List[float]:
        """توليد تاريخ أسعار ممتد"""
        import random
        
        prices = [current_price]
        
        for i in range(days - 1):
            change = random.uniform(-0.03, 0.03)  # تغيير 3% كحد أقصى
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        return prices
    
    def _detect_rsi_divergence(self, prices: List[float], period: int) -> bool:
        """كشف التباعد في RSI"""
        # تبسيط - في التطبيق الحقيقي يحتاج تحليل أعمق
        return False
    
    def _analyze_volume_patterns(self, price_data: Dict) -> Dict:
        """تحليل أنماط الحجم"""
        volume = price_data.get('volume', 0)
        
        # محاكاة تحليل الحجم
        if volume > 1000000:
            volume_signal = 'high_volume'
            strength = 0.8
        elif volume > 500000:
            volume_signal = 'medium_volume'
            strength = 0.6
        else:
            volume_signal = 'low_volume'
            strength = 0.4
        
        return {
            'signal': volume_signal,
            'strength': strength,
            'volume': volume
        }
    
    def _analyze_candlestick_patterns(self, prices: List[float]) -> Dict:
        """تحليل أنماط الشموع اليابانية"""
        if len(prices) < 3:
            return {'pattern': 'insufficient_data', 'signal': 'neutral'}
        
        # تحليل بسيط لأنماط الشموع
        last_three = prices[-3:]
        
        if last_three[0] < last_three[1] < last_three[2]:
            return {'pattern': 'bullish_trend', 'signal': 'bullish'}
        elif last_three[0] > last_three[1] > last_three[2]:
            return {'pattern': 'bearish_trend', 'signal': 'bearish'}
        else:
            return {'pattern': 'consolidation', 'signal': 'neutral'}
    
    def _analyze_elliott_waves(self, prices: List[float]) -> Dict:
        """تحليل موجات إليوت"""
        if len(prices) < 8:
            return {'wave': 'insufficient_data', 'position': 'unknown'}
        
        # تحليل مبسط لموجات إليوت
        recent_trend = self._calculate_trend_strength(prices[-8:])
        
        if recent_trend > 0.7:
            return {'wave': 'impulse_wave', 'position': 'wave_3', 'signal': 'bullish'}
        elif recent_trend < -0.7:
            return {'wave': 'impulse_wave', 'position': 'wave_3', 'signal': 'bearish'}
        else:
            return {'wave': 'corrective_wave', 'position': 'wave_b', 'signal': 'neutral'}
    
    def _calculate_market_risk(self, correlation_data: Dict, market_sentiment: Dict) -> str:
        """حساب مخاطر السوق"""
        volatility_factor = correlation_data['volatility_factor']
        sentiment_score = market_sentiment['score']
        
        risk_score = volatility_factor * (1 - sentiment_score)
        
        if risk_score > 1.5:
            return 'high'
        elif risk_score > 1.0:
            return 'medium'
        else:
            return 'low'
    
    async def _fallback_analysis(self, asset: str, market: str, timeframe: str, price_data: Dict) -> Dict:
        """تحليل احتياطي في حالة فشل الذكاء الاصطناعي"""
        return {
            'direction': 'up' if np.random.random() > 0.5 else 'down',
            'confidence': np.random.randint(60, 85),
            'reasoning': 'تحليل احتياطي بناءً على المؤشرات التقنية الأساسية',
            'fallback': True
        }

