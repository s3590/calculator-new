"""
مؤشرات تقنية متقدمة لتحليل الأسعار والتنبؤ
"""
import math
import random
from datetime import datetime, timedelta

class TechnicalIndicators:
    """فئة المؤشرات التقنية"""
    
    @staticmethod
    def rsi(prices, period=14):
        """مؤشر القوة النسبية (RSI)"""
        if len(prices) < period + 1:
            return 50  # قيمة افتراضية
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """مؤشر MACD"""
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        # حساب المتوسطات المتحركة الأسية
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # حساب خط الإشارة (متوسط متحرك أسي لخط MACD)
        macd_values = [macd_line] * signal  # تبسيط للحساب
        signal_line = sum(macd_values) / len(macd_values)
        
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def ema(prices, period):
        """المتوسط المتحرك الأسي"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    @staticmethod
    def sma(prices, period):
        """المتوسط المتحرك البسيط"""
        if len(prices) < period:
            return sum(prices) / len(prices)
        return sum(prices[-period:]) / period
    
    @staticmethod
    def bollinger_bands(prices, period=20, std_dev=2):
        """نطاقات بولينجر"""
        if len(prices) < period:
            sma = sum(prices) / len(prices)
            return {'upper': sma * 1.02, 'middle': sma, 'lower': sma * 0.98}
        
        sma = TechnicalIndicators.sma(prices, period)
        
        # حساب الانحراف المعياري
        variance = sum((price - sma) ** 2 for price in prices[-period:]) / period
        std = math.sqrt(variance)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def stochastic(highs, lows, closes, k_period=14, d_period=3):
        """مؤشر ستوكاستيك"""
        if len(closes) < k_period:
            return {'k': 50, 'd': 50}
        
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # تبسيط حساب %D
        d_percent = k_percent  # في التطبيق الحقيقي، يجب حساب متوسط %K
        
        return {'k': k_percent, 'd': d_percent}

class PredictionEngine:
    """محرك التنبؤ المتقدم"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def generate_price_history(self, base_price, days=30):
        """توليد تاريخ أسعار وهمي للاختبار"""
        prices = [base_price]
        highs = [base_price * 1.01]
        lows = [base_price * 0.99]
        
        for i in range(days - 1):
            # محاكاة تحرك السعر
            change_percent = random.uniform(-0.05, 0.05)  # تغيير بنسبة 5% كحد أقصى
            new_price = prices[-1] * (1 + change_percent)
            
            # حساب الأعلى والأدنى
            high = new_price * random.uniform(1.001, 1.02)
            low = new_price * random.uniform(0.98, 0.999)
            
            prices.append(new_price)
            highs.append(high)
            lows.append(low)
        
        return prices, highs, lows
    
    def analyze_asset(self, asset, market, timeframe, current_price=None):
        """تحليل شامل للأصل"""
        if current_price is None:
            current_price = self.get_base_price(asset, market)
        
        # توليد بيانات تاريخية
        prices, highs, lows = self.generate_price_history(current_price)
        
        # حساب المؤشرات التقنية
        rsi = self.indicators.rsi(prices)
        macd = self.indicators.macd(prices)
        bollinger = self.indicators.bollinger_bands(prices)
        stochastic = self.indicators.stochastic(highs, lows, prices)
        
        # تحليل الاتجاه
        trend_analysis = self.analyze_trend(prices)
        
        # تحليل الحجم (محاكاة)
        volume_analysis = self.analyze_volume(asset, market)
        
        # معنويات السوق
        market_sentiment = self.analyze_market_sentiment(market, timeframe)
        
        return {
            'rsi': rsi,
            'macd': macd,
            'bollinger': bollinger,
            'stochastic': stochastic,
            'trend': trend_analysis,
            'volume': volume_analysis,
            'sentiment': market_sentiment,
            'prices': prices[-10:]  # آخر 10 أسعار
        }
    
    def predict_direction(self, analysis, timeframe):
        """التنبؤ بالاتجاه بناءً على التحليل"""
        signals = []
        weights = []
        
        # إشارات RSI
        if analysis['rsi'] < 30:
            signals.append('up')
            weights.append(0.8)  # إشارة قوية للشراء
        elif analysis['rsi'] > 70:
            signals.append('down')
            weights.append(0.8)  # إشارة قوية للبيع
        else:
            signals.append('neutral')
            weights.append(0.3)
        
        # إشارات MACD
        if analysis['macd']['histogram'] > 0:
            signals.append('up')
            weights.append(0.7)
        else:
            signals.append('down')
            weights.append(0.7)
        
        # إشارات نطاقات بولينجر
        current_price = analysis['prices'][-1]
        if current_price < analysis['bollinger']['lower']:
            signals.append('up')
            weights.append(0.6)
        elif current_price > analysis['bollinger']['upper']:
            signals.append('down')
            weights.append(0.6)
        
        # إشارات ستوكاستيك
        if analysis['stochastic']['k'] < 20:
            signals.append('up')
            weights.append(0.5)
        elif analysis['stochastic']['k'] > 80:
            signals.append('down')
            weights.append(0.5)
        
        # تحليل الاتجاه
        signals.append(analysis['trend']['direction'])
        weights.append(analysis['trend']['strength'])
        
        # تحليل الحجم
        signals.append(analysis['volume']['signal'])
        weights.append(analysis['volume']['strength'])
        
        # معنويات السوق
        signals.append(analysis['sentiment']['direction'])
        weights.append(analysis['sentiment']['strength'])
        
        # حساب النتيجة النهائية
        up_score = sum(w for s, w in zip(signals, weights) if s == 'up')
        down_score = sum(w for s, w in zip(signals, weights) if s == 'down')
        
        # تعديل النتيجة بناءً على الإطار الزمني
        timeframe_factor = self.get_timeframe_factor(timeframe)
        
        final_up_score = up_score * timeframe_factor
        final_down_score = down_score * timeframe_factor
        
        # تحديد الاتجاه والثقة
        if final_up_score > final_down_score:
            direction = 'up'
            confidence = min(95, max(55, int((final_up_score / (final_up_score + final_down_score)) * 100)))
        else:
            direction = 'down'
            confidence = min(95, max(55, int((final_down_score / (final_up_score + final_down_score)) * 100)))
        
        return {
            'direction': direction,
            'confidence': confidence,
            'factors': {
                'technical_analysis': min(1.0, (up_score + down_score) / 6),
                'market_sentiment': analysis['sentiment']['strength'],
                'volume_analysis': analysis['volume']['strength'],
                'trend_analysis': analysis['trend']['strength']
            },
            'signals': {
                'rsi': analysis['rsi'],
                'macd_signal': 'bullish' if analysis['macd']['histogram'] > 0 else 'bearish',
                'bollinger_position': self.get_bollinger_position(current_price, analysis['bollinger']),
                'stochastic_signal': 'oversold' if analysis['stochastic']['k'] < 20 else 'overbought' if analysis['stochastic']['k'] > 80 else 'neutral'
            }
        }
    
    def get_base_price(self, asset, market):
        """الحصول على السعر الأساسي للأصل"""
        base_prices = {
            'العملات': {
                'EUR/USD': 1.0850,
                'GBP/USD': 1.2650,
                'USD/JPY': 149.50
            },
            'العملات المشفرة': {
                'BTC/USD': 43000,
                'ETH/USD': 2600
            },
            'السلع': {
                'ذهب': 2050,
                'فضة': 24.50,
                'نفط': 78.50
            },
            'الأسهم': {
                'AAPL': 175,
                'TSLA': 240,
                'AMZN': 145
            }
        }
        
        return base_prices.get(market, {}).get(asset, 100)
    
    def analyze_trend(self, prices):
        """تحليل الاتجاه"""
        if len(prices) < 10:
            return {'direction': 'neutral', 'strength': 0.5}
        
        # حساب المتوسطات المتحركة
        short_ma = sum(prices[-5:]) / 5
        long_ma = sum(prices[-10:]) / 10
        
        if short_ma > long_ma * 1.01:
            return {'direction': 'up', 'strength': 0.8}
        elif short_ma < long_ma * 0.99:
            return {'direction': 'down', 'strength': 0.8}
        else:
            return {'direction': 'neutral', 'strength': 0.4}
    
    def analyze_volume(self, asset, market):
        """تحليل الحجم (محاكاة)"""
        # محاكاة تحليل الحجم
        volume_strength = random.uniform(0.4, 0.9)
        volume_signal = 'up' if random.random() > 0.5 else 'down'
        
        return {
            'signal': volume_signal,
            'strength': volume_strength
        }
    
    def analyze_market_sentiment(self, market, timeframe):
        """تحليل معنويات السوق"""
        # محاكاة معنويات السوق بناءً على نوع السوق
        sentiment_map = {
            'العملات': {'direction': 'up', 'strength': 0.6},
            'العملات المشفرة': {'direction': 'up', 'strength': 0.7},
            'السلع': {'direction': 'down', 'strength': 0.5},
            'الأسهم': {'direction': 'up', 'strength': 0.8}
        }
        
        base_sentiment = sentiment_map.get(market, {'direction': 'neutral', 'strength': 0.5})
        
        # تعديل بناءً على الإطار الزمني
        if timeframe in ['1 دقيقة', '5 دقائق']:
            base_sentiment['strength'] *= 0.8  # أقل موثوقية للإطارات القصيرة
        
        return base_sentiment
    
    def get_timeframe_factor(self, timeframe):
        """عامل تعديل الإطار الزمني"""
        factors = {
            '1 دقيقة': 0.7,
            '5 دقائق': 0.8,
            '10 دقائق': 0.9,
            '30 دقيقة': 0.95,
            '1 ساعة': 1.0
        }
        return factors.get(timeframe, 1.0)
    
    def get_bollinger_position(self, price, bollinger):
        """تحديد موقع السعر في نطاقات بولينجر"""
        if price > bollinger['upper']:
            return 'above_upper'
        elif price < bollinger['lower']:
            return 'below_lower'
        elif price > bollinger['middle']:
            return 'above_middle'
        else:
            return 'below_middle'

