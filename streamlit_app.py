import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import pytz
from enum import Enum
import plotly.graph_objects as go
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Configuration and Constants
@dataclass
class ImageFormatConfig:
    extension: str
    compression_ratio: float
    base64_overhead: float

class ImageFormat(Enum):
    """
    Enum for supported image formats with their characteristics.
    The compression_ratio represents typical compression achieved for photo content.
    Base64 encoding increases size by approximately 1.37x due to 64/48 byte ratio.
    """
    JPEG = ImageFormatConfig('.jpg', 0.3, 1.37)
    PNG = ImageFormatConfig('.png', 0.7, 1.37)
    BASE64 = ImageFormatConfig('.b64', 1.0, 1.37)

class ServerConfig:
    """Configuration constants for server calculations"""
    PEAK_START_HOUR = 17
    PEAK_END_HOUR = 2
    SESSION_INTERVAL = 20  # seconds
    PEAK_LOAD_PERCENTAGE = 0.7  # 70% of daily users during peak
    SPIKE_BUFFER = 1.3  # 30% buffer for random spikes
    NETWORK_OVERHEAD = 1.2  # 20% overhead for WebSocket headers
    MEMORY_PER_PHOTO_MB = 200
    CPU_CORES_PER_100_PHOTOS = 2
    PNG_CPU_MULTIPLIER = 1.5

class ServerLoadCalculator:
    """Calculate server resource requirements based on user activity and image processing needs"""
    
    def __init__(
        self,
        image_format: str = 'JPEG',
        raw_image_size_kb: int = 500,
        daily_active_users: int = 4000,
        frames_per_second: int = 3,
        photos_per_user: int = 3,
        max_session_duration: int = 5
    ):
        """
        Initialize calculator with configurable parameters
        
        Args:
            image_format: Format of the images (JPEG, PNG, BASE64)
            raw_image_size_kb: Size of raw images in KB
            daily_active_users: Number of daily active users
            frames_per_second: Number of frames processed per second
            photos_per_user: Number of photos per user session
            max_session_duration: Maximum session duration in minutes
        """
        self.validate_inputs(
            raw_image_size_kb,
            daily_active_users,
            frames_per_second,
            photos_per_user,
            max_session_duration
        )
        
        self.image_format = ImageFormat[image_format.upper()]
        self.raw_image_size_kb = raw_image_size_kb
        self.daily_active_users = daily_active_users
        self.frames_per_second = frames_per_second
        self.photos_per_user = photos_per_user
        self.max_session_duration = max_session_duration * 60  # Convert to seconds
        
        self.peak_duration = (
            24 - ServerConfig.PEAK_START_HOUR + ServerConfig.PEAK_END_HOUR
        )
        self.actual_image_size = self._calculate_actual_image_size()

    @staticmethod
    def validate_inputs(
        raw_image_size_kb: int,
        daily_active_users: int,
        frames_per_second: int,
        photos_per_user: int,
        max_session_duration: int
    ) -> None:
        """Validate input parameters"""
        if raw_image_size_kb <= 0:
            raise ValueError("Raw image size must be positive")
        if daily_active_users <= 0:
            raise ValueError("Daily active users must be positive")
        if frames_per_second <= 0:
            raise ValueError("Frames per second must be positive")
        if photos_per_user <= 0:
            raise ValueError("Photos per user must be positive")
        if max_session_duration <= 0:
            raise ValueError("Session duration must be positive")

    def _calculate_actual_image_size(self) -> float:
        """Calculate actual image size with compression and encoding overhead"""
        format_params = self.image_format.value
        compressed_size = self.raw_image_size_kb * format_params.compression_ratio
        
        if self.image_format == ImageFormat.BASE64:
            return compressed_size * format_params.base64_overhead
        return compressed_size

    def calculate_peak_concurrent_users(self) -> int:
        """Calculate maximum concurrent users during peak hours"""
        total_session_time = (
            self.max_session_duration + ServerConfig.SESSION_INTERVAL
        ) * self.photos_per_user
        users_per_hour = 3600 / total_session_time
        
        peak_hours_users = (
            self.daily_active_users * ServerConfig.PEAK_LOAD_PERCENTAGE
        )
        avg_concurrent_users = peak_hours_users / (users_per_hour * self.peak_duration)
        
        return int(np.ceil(avg_concurrent_users * ServerConfig.SPIKE_BUFFER))

    def calculate_photos_per_minute(self) -> int:
        """Calculate maximum photos processed per minute during peak load"""
        peak_concurrent_users = self.calculate_peak_concurrent_users()
        total_frames_per_second = peak_concurrent_users * self.frames_per_second
        return int(np.ceil(total_frames_per_second * 60))

    def calculate_bandwidth_requirements(self) -> float:
        """Calculate network bandwidth requirements in Mbps"""
        photos_per_minute = self.calculate_photos_per_minute()
        bytes_per_second = (photos_per_minute * self.actual_image_size * 1024) / 60
        mbps = (bytes_per_second * 8) / (1024 * 1024)
        return mbps * ServerConfig.NETWORK_OVERHEAD

    def calculate_storage_requirements(self) -> Dict[str, float]:
        """Calculate storage requirements in GB per hour and day"""
        photos_per_minute = self.calculate_photos_per_minute()
        hourly_storage_gb = (
            photos_per_minute * 60 * self.actual_image_size
        ) / (1024 * 1024)
        
        return {
            'hourly_gb': hourly_storage_gb,
            'daily_gb': hourly_storage_gb * self.peak_duration
        }

    def estimate_server_resources(self) -> Dict[str, int]:
        """Estimate required server resources"""
        peak_concurrent_users = self.calculate_peak_concurrent_users()
        photos_per_minute = self.calculate_photos_per_minute()
        
        format_cpu_multiplier = (
            ServerConfig.PNG_CPU_MULTIPLIER 
            if self.image_format == ImageFormat.PNG 
            else 1.0
        )
        
        required_memory_gb = (
            peak_concurrent_users * ServerConfig.MEMORY_PER_PHOTO_MB
        ) / 1024
        required_cpu_cores = np.ceil(
            (photos_per_minute / 100) 
            * ServerConfig.CPU_CORES_PER_100_PHOTOS 
            * format_cpu_multiplier
        )
        
        return {
            'required_memory_gb': int(np.ceil(required_memory_gb)),
            'required_cpu_cores': int(required_cpu_cores),
            'recommended_servers': int(np.ceil(required_cpu_cores / 32))
        }

def create_hourly_load_chart(calculator: ServerLoadCalculator) -> go.Figure:
    """Create interactive chart showing estimated hourly load distribution"""
    hours = list(range(24))
    peak_users = calculator.calculate_peak_concurrent_users()
    
    load_distribution = [
        peak_users if (
            ServerConfig.PEAK_START_HOUR <= hour 
            or hour < ServerConfig.PEAK_END_HOUR
        ) else peak_users * 0.3
        for hour in hours
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=load_distribution,
        mode='lines+markers',
        name='Concurrent Users',
        hovertemplate='Hour: %{x}<br>Users: %{y:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Estimated Hourly Server Load',
        xaxis_title='Hour of Day (MSK)',
        yaxis_title='Concurrent Users',
        height=400,
        template='plotly_white'
    )
    
    return fig

def create_sidebar_inputs() -> Dict[str, Any]:
    """Create and handle sidebar inputs"""
    st.sidebar.title("Configuration")
    
    with st.sidebar.expander("📊 Basic Parameters", expanded=True):
        daily_users = st.number_input(
            "Daily Active Users",
            min_value=100,
            max_value=100000,
            value=4000,
            help="Expected number of unique users per day"
        )
        
        image_format = st.selectbox(
            "Image Format",
            ["JPEG", "PNG", "BASE64"],
            help="Select the image format for processing"
        )
        
        raw_image_size = st.number_input(
            "Raw Image Size (KB)",
            min_value=1,
            max_value=10000,
            value=5,
            help="Size of uncompressed images in kilobytes",
            step=1
        )
    
    with st.sidebar.expander("⚙️ Advanced Parameters", expanded=True):
        frames_per_second = st.slider(
            "Frames per Second",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of frames processed per second"
        )
        
        photos_per_user = st.slider(
            "Photos per User",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of photos each user processes"
        )
        
        session_duration = st.slider(
            "Max Session Duration (minutes)",
            min_value=1,
            max_value=15,
            value=5,
            help="Maximum duration of a user session"
        )
    
    return {
        'image_format': image_format,
        'raw_image_size_kb': raw_image_size,
        'daily_active_users': daily_users,
        'frames_per_second': frames_per_second,
        'photos_per_user': photos_per_user,
        'max_session_duration': session_duration
    }

def display_metrics(calculator: ServerLoadCalculator) -> None:
    """Display calculated metrics in expandable sections"""
    with st.expander("📊 Load Analysis", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Peak Concurrent Users",
                calculator.calculate_peak_concurrent_users()
            )
            st.metric(
                "Photos per Minute",
                calculator.calculate_photos_per_minute()
            )
        
        with col2:
            bandwidth = calculator.calculate_bandwidth_requirements()
            st.metric(
                "Required Bandwidth (Mbps)",
                f"{bandwidth:.2f}"
            )
            st.metric(
                "Recommended Bandwidth (Mbps)",
                f"{bandwidth * 1.5:.2f}"
            )
    
    with st.expander("💾 Storage Requirements", expanded=True):
        storage = calculator.calculate_storage_requirements()
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Hourly Storage (GB)",
                f"{storage['hourly_gb']:.2f}"
            )
        with col2:
            st.metric(
                "Daily Storage (GB)",
                f"{storage['daily_gb']:.2f}"
            )
    
    with st.expander("🖥️ Compute Resources", expanded=True):
        resources = calculator.estimate_server_resources()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Required RAM (GB)",
                resources['required_memory_gb']
            )
        with col2:
            st.metric(
                "Required CPU Cores",
                resources['required_cpu_cores']
            )
        with col3:
            st.metric(
                "Recommended Servers",
                resources['recommended_servers']
            )

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Server Load Calculator",
        page_icon="🖥️",
        layout="wide"
    )
    
    st.title("Interactive Server Load Calculator")
    st.markdown("""
    This application helps estimate server requirements for a photo processing system.
    Configure the parameters in the sidebar to see how they affect server load and resource requirements.
    """)
    
    try:
        # Get inputs from sidebar
        inputs = create_sidebar_inputs()
        
        # Initialize calculator with user inputs
        calculator = ServerLoadCalculator(**inputs)
        
        # Display results
        display_metrics(calculator)
        
        # Display hourly load distribution chart
        st.subheader("📈 Hourly Load Distribution")
        st.plotly_chart(
            create_hourly_load_chart(calculator),
            use_container_width=True
        )
        
        # Display recommendations and calculation methodology
        st.subheader("📝 Рекомендации и методология расчетов")
        
        with st.expander("🔄 Системные рекомендации", expanded=False):
            st.info("""
            На основе расчетных требований рекомендуется внедрить:
            • Систему очередей для обработки пиковых нагрузок
            • Автоматическое масштабирование на основе количества одновременных пользователей
            • CDN для глобального распределения пользователей
            • Эффективную очистку обработанных изображений
            • Мониторинг состояния WebSocket-соединений
            • Ограничение скорости для каждого пользователя
            """)
        
        with st.expander("📊 Методология расчетов", expanded=True):
            st.markdown("""
            ### Логика и допущения при расчетах
            
            #### 1. Расчет пиковых одновременных пользователей
            
            **Исходные данные и допущения:**
            - Пиковые часы: с 17:00 до 02:00 (9 часов)
            - 70% дневных пользователей приходится на пиковые часы
            - Добавляется 30% буфер для случайных всплесков
            
            **Формула расчета:**
            1. Время сессии = (Длительность сессии + 20 секунд интервал) × Количество фото
            2. Пользователей в час = 3600 / Время сессии
            3. Пиковые пользователи = (DAU × 0.7) / (Пользователей в час × 9)
            4. Финальное значение = Пиковые пользователи × 1.3 (буфер)
            
            #### 2. Расчет требований к полосе пропускания
            
            **Исходные данные и допущения:**
            - Учитывается сжатие в зависимости от формата (JPEG: 0.3, PNG: 0.7, BASE64: 1.0)
            - Добавляется 20% на накладные расходы WebSocket
            - Для BASE64 учитывается увеличение размера на 37%
            
            **Формула расчета:**
            1. Сжатый размер = Исходный размер × Коэффициент сжатия
            2. Байт в секунду = (Фото в минуту × Размер × 1024) / 60
            3. Мбит/с = (Байт в секунду × 8) / (1024 × 1024) × 1.2
            
            #### 3. Расчет требований к хранилищу
            
            **Исходные данные и допущения:**
            - Хранение всех фото в течение пиковых часов
            - Учитывается только временное хранение во время обработки
            
            **Формула расчета:**
            1. Хранилище в час = (Фото в минуту × 60 × Размер) / (1024 × 1024)
            2. Суточное хранилище = Хранилище в час × 9 (пиковые часы)
            
            #### 4. Расчет вычислительных ресурсов
            
            **Исходные данные и допущения:**
            - 200 МБ RAM на обработку одного фото
            - 2 ядра CPU на каждые 100 фото в минуту
            - Для PNG требуется на 50% больше CPU
            - Сервер поддерживает до 32 ядер
            
            **Формула расчета:**
            1. RAM (GB) = (Пиковые пользователи × 200 МБ) / 1024
            2. CPU ядра = (Фото в минуту / 100) × 2 × (1.5 для PNG)
            3. Количество серверов = Округление вверх (CPU ядра / 32)
            """)
            
        with st.expander("⚠️ Ограничения и предупреждения", expanded=False):
            st.warning("""
            **Важно учитывать следующие ограничения расчетов:**
            
            1. Распределение пользователей предполагается по московскому времени
            2. Не учитывается географическое распределение пользователей
            3. Предполагается линейная зависимость потребления ресурсов
            4. Расчеты основаны на средних показателях сжатия изображений
            5. Не учитываются особенности конкретного оборудования
            6. Реальная производительность может отличаться от расчетной
            
            Рекомендуется провести нагрузочное тестирование для уточнения расчетов.
            """)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()