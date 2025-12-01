"""
Keysight E36313A Power Supply Driver
VISA protocol을 이용한 TCPIP 통신 드라이버
"""
import pyvisa
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ChannelData:
    """채널 측정 데이터"""
    channel: int
    voltage: float
    current: float
    timestamp: float


class KeysightE36313ADriver:
    """Keysight E36313A 전원공급장치 드라이버"""
    
    # 채널별 최대값 (E36313A 기준)
    MAX_VOLTAGE = {1: 6.00, 2: 25.00, 3: 25.00}
    MAX_CURRENT = {1: 10.0, 2: 2.00, 3: 2.00}
    
    def __init__(self, ip_address: str, timeout: int = 5000):
        """
        Args:
            ip_address: 장비 IP 주소
            timeout: 통신 타임아웃 (ms)
        """
        self.ip_address = ip_address
        self.timeout = timeout
        self.instrument: Optional[pyvisa.Resource] = None
        self.rm: Optional[pyvisa.ResourceManager] = None
        self._connected = False
        
    def connect(self) -> bool:
        """장비 연결"""
        try:
            self.rm = pyvisa.ResourceManager('@py')
            resource_name = f"TCPIP::{self.ip_address}::5025::SOCKET"
            self.instrument = self.rm.open_resource(resource_name)
            self.instrument.timeout = self.timeout

            self.instrument.write_termination = "\n"
            self.instrument.read_termination = "\n"
            
            # 연결 확인
            idn = self.instrument.query("*IDN?")
            logger.info(f"Connected to: {idn.strip()}")
            
            # 초기화
            self.instrument.write("*CLS")  # 에러 큐 클리어
            self.instrument.write("*RST")  # 리셋
            time.sleep(0.5)
            
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            return False
    
    def disconnect(self):
        """장비 연결 해제"""
        try:
            if self.instrument:
                # 모든 채널 출력 OFF
                self.set_all_outputs(False)
                self.instrument.close()
            if self.rm:
                self.rm.close()
            self._connected = False
            logger.info("Disconnected successfully")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")
    
    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self._connected
    
    def check_errors(self) -> List[str]:
        """에러 큐 확인"""
        errors = []
        try:
            while True:
                error = self.instrument.query("SYST:ERR?").strip()
                if error.startswith('+0,"No error"'):
                    break
                errors.append(error)
        except Exception as e:
            logger.error(f"Error checking failed: {e}")
        return errors
    
    # ========== 측정 함수 ==========
    
    def measure_voltage(self, channel: int) -> Optional[float]:
        """개별 채널 전압 측정
        
        Args:
            channel: 채널 번호 (1-3)
            
        Returns:
            측정 전압 (V)
        """
        try:
            if not 1 <= channel <= 3:
                raise ValueError(f"Invalid channel: {channel}")
            
            voltage = float(self.instrument.query(f"MEAS:VOLT? (@{channel})"))
            return voltage
            
        except Exception as e:
            logger.error(f"Voltage measurement error (CH{channel}): {e}")
            return None
    
    def measure_current(self, channel: int) -> Optional[float]:
        """개별 채널 전류 측정
        
        Args:
            channel: 채널 번호 (1-3)
            
        Returns:
            측정 전류 (A)
        """
        try:
            if not 1 <= channel <= 3:
                raise ValueError(f"Invalid channel: {channel}")
            
            current = float(self.instrument.query(f"MEAS:CURR? (@{channel})"))
            return current
            
        except Exception as e:
            logger.error(f"Current measurement error (CH{channel}): {e}")
            return None
    '''
    def measure_channel(self, channel: int) -> Optional[ChannelData]:
        """개별 채널 전압/전류 동시 측정
        
        Args:
            channel: 채널 번호 (1-3)
            
        Returns:
            ChannelData 객체
        """
        try:
            voltage = self.measure_voltage(channel)
            current = self.measure_current(channel)
            
            if voltage is not None and current is not None:
                return ChannelData(
                    channel=channel,
                    voltage=voltage,
                    current=current,
                    timestamp=time.time()
                )
            return None
            
        except Exception as e:
            logger.error(f"Channel measurement error (CH{channel}): {e}")
            return None
    
    def measure_all_channels(self) -> Dict[int, ChannelData]:
        """전체 채널 일괄 측정
        
        Returns:
            채널별 측정 데이터 딕셔너리
        """
        results = {}
        timestamp = time.time()
        
        try:
            # 채널 리스트 형태로 일괄 측정 (성능 최적화)
            voltages = self.instrument.query("MEAS:VOLT? CH{ch}").strip().split(',')
            currents = self.instrument.query("MEAS:CURR? CH{ch}").strip().split(',')
            
            for ch in range(1, 4):
                try:
                    results[ch] = ChannelData(
                        channel=ch,
                        voltage=float(voltages[ch-1]),
                        current=float(currents[ch-1]),
                        timestamp=timestamp
                    )
                except (IndexError, ValueError) as e:
                    logger.error(f"Parse error for CH{ch}: {e}")
                    
        except Exception as e:
            logger.error(f"Batch measurement error: {e}")
            # Fallback: 개별 측정
            for ch in range(1, 4):
                data = self.measure_channel(ch)
                if data:
                    results[ch] = data
        
        return results
    '''

    def measure_channel(self, channels: List[int]) -> Dict[int, ChannelData]:
        """지정된 채널들 일괄 측정 (최적화 버전)
        
        Args:
            channels: 측정할 채널 번호 리스트 (예: [1, 2, 3])
            
        Returns:
            채널별 측정 데이터 딕셔너리
        """
        results = {}
        timestamp = time.time()
        
        if not channels:
            return results
        
        try:
            # 채널 리스트 형태로 일괄 측정
            channels_str = ','.join(str(ch) for ch in sorted(channels))
            
            # write/read 분리로 약간의 성능 향상
            self.instrument.write(f"MEAS:VOLT? (@{channels_str})")
            voltage_response = self.instrument.read().strip()
            
            self.instrument.write(f"MEAS:CURR? (@{channels_str})")
            current_response = self.instrument.read().strip()
            
            voltages = voltage_response.split(',')
            currents = current_response.split(',')
            
            for idx, ch in enumerate(sorted(channels)):
                try:
                    results[ch] = ChannelData(
                        channel=ch,
                        voltage=float(voltages[idx]),
                        current=float(currents[idx]),
                        timestamp=timestamp
                    )
                except (IndexError, ValueError) as e:
                    logger.error(f"Parse error for CH{ch}: {e}")
                    
        except Exception as e:
            logger.error(f"Batch measurement error: {e}")
        
        return results

    def measure_all_channel(self) -> Dict[int, ChannelData]:
        """전체 채널 일괄 측정
        
        Returns:
            채널별 측정 데이터 딕셔너리
        """
        return self.measure_channel([1, 2, 3])
    
    # ========== 설정 함수 ==========
    
    def set_voltage(self, channel: int, voltage: float) -> bool:
        """개별 채널 전압 설정
        
        Args:
            channel: 채널 번호 (1-3)
            voltage: 설정 전압 (V)
            
        Returns:
            성공 여부
        """
        try:
            if not 1 <= channel <= 3:
                raise ValueError(f"Invalid channel: {channel}")
            
            if not 0 <= voltage <= self.MAX_VOLTAGE[channel]:
                raise ValueError(
                    f"Voltage out of range. Max: {self.MAX_VOLTAGE[channel]}V"
                )
            
            self.instrument.write(f"VOLT {voltage}, (@{channel})")
            time.sleep(0.1)
            
            # 설정 확인
            set_voltage = float(self.instrument.query(f"VOLT? (@{channel})"))
            if abs(set_voltage - voltage) < 0.01:
                logger.info(f"CH{channel} voltage set to {voltage}V")
                return True
            else:
                logger.warning(
                    f"CH{channel} voltage mismatch: set={voltage}, read={set_voltage}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Voltage setting error (CH{channel}): {e}")
            return False
    
    def set_current(self, channel: int, current: float) -> bool:
        """개별 채널 전류 제한 설정
        
        Args:
            channel: 채널 번호 (1-3)
            current: 설정 전류 (A)
            
        Returns:
            성공 여부
        """
        try:
            if not 1 <= channel <= 3:
                raise ValueError(f"Invalid channel: {channel}")
            
            if not 0 <= current <= self.MAX_CURRENT[channel]:
                raise ValueError(
                    f"Current out of range. Max: {self.MAX_CURRENT[channel]}A"
                )
            
            self.instrument.write(f"CURR {current}, (@{channel})")
            time.sleep(0.1)
            
            # 설정 확인
            set_current = float(self.instrument.query(f"CURR? (@{channel})"))
            if abs(set_current - current) < 0.001:
                logger.info(f"CH{channel} current limit set to {current}A")
                return True
            else:
                logger.warning(
                    f"CH{channel} current mismatch: set={current}, read={set_current}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Current setting error (CH{channel}): {e}")
            return False
    
    def set_output(self, channel: int, state: bool) -> bool:
        """개별 채널 출력 ON/OFF
        
        Args:
            channel: 채널 번호 (1-3)
            state: True=ON, False=OFF
            
        Returns:
            성공 여부
        """
        try:
            if not 1 <= channel <= 3:
                raise ValueError(f"Invalid channel: {channel}")
            
            cmd = "ON" if state else "OFF"
            self.instrument.write(f"OUTP {cmd}, (@{channel})")
            time.sleep(0.1)
            
            # 상태 확인
            status = int(self.instrument.query(f"OUTP? (@{channel})"))
            success = (status == 1) if state else (status == 0)
            
            if success:
                logger.info(f"CH{channel} output {cmd}")
            return success
            
        except Exception as e:
            logger.error(f"Output control error (CH{channel}): {e}")
            return False
    
    def set_all_outputs(self, state: bool) -> Dict[int, bool]:
        """전체 채널 출력 일괄 ON/OFF
        
        Args:
            state: True=ON, False=OFF
            
        Returns:
            채널별 성공 여부 딕셔너리
        """
        results = {}
        for ch in range(1, 4):
            results[ch] = self.set_output(ch, state)
        return results
    
    # ========== 상태 확인 ==========
    
    def get_output_status(self, channel: int) -> Optional[bool]:
        """채널 출력 상태 확인
        
        Args:
            channel: 채널 번호 (1-3)
            
        Returns:
            True=ON, False=OFF, None=에러
        """
        try:
            status = int(self.instrument.query(f"OUTP? (@{channel})"))
            return status == 1
        except Exception as e:
            logger.error(f"Status check error (CH{channel}): {e}")
            return None
    
    def get_all_status(self) -> Dict[int, bool]:
        """전체 채널 출력 상태 확인"""
        return {
            ch: self.get_output_status(ch) 
            for ch in range(1, 4)
        }