from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set
import asyncio
import json
from datetime import datetime
from contextlib import asynccontextmanager
import sqlite3
from pathlib import Path
import os

from driver.keysight_driver import KeysightE36313ADriver, ChannelData

# ========== 전역 변수 ==========
driver: Optional[KeysightE36313ADriver] = None
db_path = Path(os.getenv("DATABASE_PATH", "logs.db"))

# 활성 WebSocket 연결 추적
active_channels: Set[int] = set()

# 로깅 활성화 설정
logging_enabled: Dict[int, bool] = {1: True, 2: True, 3: True}


# ========== Database 초기화 ==========
def init_database():
    """SQLite 데이터베이스 초기화 - 채널별 테이블 생성"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 채널 1, 2, 3에 대해 각각 독립된 테이블 생성
    for channel in [1, 2, 3]:
        table_name = f"channel_{channel}"
        
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                datetime TEXT NOT NULL,
                voltage REAL NOT NULL,
                current REAL NOT NULL,
                power REAL NOT NULL
            )
        """)
        
        # 타임스탬프 인덱스
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp 
            ON {table_name}(timestamp)
        """)
        
        print(f"Table '{table_name}' initialized")
    
    conn.commit()
    conn.close()


def save_to_database(data: ChannelData):
    """측정 데이터를 채널별 테이블에 저장"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        table_name = f"channel_{data.channel}"
        
        cursor.execute(f"""
            INSERT INTO {table_name}
            (timestamp, datetime, voltage, current, power)
            VALUES (?, ?, ?, ?, ?)
        """, (
            data.timestamp,
            datetime.fromtimestamp(data.timestamp).isoformat(),
            data.voltage,
            data.current,
            data.voltage * data.current
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Database save error (Channel {data.channel}): {e}")


# ========== Lifespan 관리 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    # Startup
    init_database()
    print("Database initialized with channel-specific tables")
    yield
    # Shutdown
    if driver and driver.is_connected():
        driver.disconnect()
    print("Application shutdown")


# ========== FastAPI 앱 초기화 ==========
app = FastAPI(
    title="Keysight E36313A Control API",
    description="Power Supply Control Backend with Channel-Specific Tables",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Pydantic Models ==========
class ConnectionConfig(BaseModel):
    ip_address: str = Field(..., example="192.168.1.100")
    timeout: int = Field(5000, ge=1000, le=30000)


class voltageSetting(BaseModel):
    voltage: float = Field(..., ge=0)
    
class currentSetting(BaseModel):
    current: float = Field(..., ge=0)

class LoggingConfig(BaseModel):
    channel: int = Field(..., ge=1, le=3)
    enabled: bool

class MeasurementResponse(BaseModel):
    channel: int
    voltage: float
    current: float
    power: float
    timestamp: float
    datetime: str


class StatusResponse(BaseModel):
    connected: bool
    outputs: Dict[int, bool]


# ========== API Endpoints ==========

@app.get("/")
async def root():
    """API 정보"""
    return {
        "name": "Keysight E36313A Control API",
        "version": "1.0.0",
        "status": "running",
        "database_structure": "channel-specific tables (channel_1, channel_2, channel_3)"
    }


@app.post("/connect")
async def connect(config: ConnectionConfig):
    """장비 연결"""
    global driver
    
    try:
        if driver and driver.is_connected():
            return {"status": "already_connected", "ip": driver.ip_address}
        
        driver = KeysightE36313ADriver(
            ip_address=config.ip_address,
            timeout=config.timeout
        )
        
        success = driver.connect()
        
        if success:
            return {
                "status": "connected",
                "ip": config.ip_address,
                "message": "Successfully connected"
            }
        else:
            raise HTTPException(status_code=500, detail="Connection failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/disconnect")
async def disconnect():
    """장비 연결 해제"""
    global driver
    
    if not driver:
        raise HTTPException(status_code=400, detail="Not connected")
    
    try:
        driver.disconnect()
        driver = None
        return {"status": "disconnected", "message": "Successfully disconnected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """연결 및 출력 상태 확인"""
    if not driver or not driver.is_connected():
        return StatusResponse(connected=False, outputs={})
    
    try:
        outputs = driver.get_all_status()
        return StatusResponse(connected=True, outputs=outputs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 측정 API ==========

@app.get("/measure/{channel}", response_model=MeasurementResponse)
async def measure_channel(channel: int):
    """개별 채널 측정"""
    if not driver or not driver.is_connected():
        raise HTTPException(status_code=400, detail="Not connected")
    
    if not 1 <= channel <= 3:
        raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
    
    try:
        results = driver.measure_channel([channel])
        
        if channel not in results:
            raise HTTPException(status_code=500, detail="Measurement failed")
        
        data = results[channel]
        
        # 채널별 테이블에 저장
        save_to_database(data)
        
        return MeasurementResponse(
            channel=data.channel,
            voltage=data.voltage,
            current=data.current,
            power=data.voltage * data.current,
            timestamp=data.timestamp,
            datetime=datetime.fromtimestamp(data.timestamp).isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 설정 API ==========

@app.post("/set/volt/{channel}")
async def set_voltage(channel: int, setting: voltageSetting):
    """개별 채널 전압 설정"""
    if not driver or not driver.is_connected():
        raise HTTPException(status_code=400, detail="Not connected")
    
    if not 1 <= channel <= 3:
        raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
    
    try:
        if setting.voltage > driver.MAX_VOLTAGE[channel]:
            raise HTTPException(
                status_code=400, 
                detail=f"Voltage exceeds max ({driver.MAX_VOLTAGE[channel]}V)"
            )
        
        success = driver.set_voltage(channel, setting.voltage)
        
        if success:
            return {
                "status": "success",
                "channel": channel,
                "voltage": setting.voltage,
            }
        else:
            raise HTTPException(status_code=500, detail="Setting failed")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/set/current/{channel}")
async def set_current(channel: int, setting: currentSetting):
    """개별 채널 전류 설정"""
    if not driver or not driver.is_connected():
        raise HTTPException(status_code=400, detail="Not connected")
    
    if not 1 <= channel <= 3:
        raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
    
    try:
        if setting.current > driver.MAX_CURRENT[channel]:
            raise HTTPException(
                status_code=400, 
                detail=f"Current exceeds max ({driver.MAX_CURRENT[channel]}A)"
            )
        
        success = driver.set_current(channel, setting.current)
        
        if success:
            return {
                "status": "success",
                "channel": channel,
                "current": setting.current,
            }
        else:
            raise HTTPException(status_code=500, detail="Setting failed")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 출력 제어 API ==========

@app.post("/output/{channel}/{state}")
async def set_output(channel: int, state: bool):
    """개별 채널 출력 ON/OFF"""
    if not driver or not driver.is_connected():
        raise HTTPException(status_code=400, detail="Not connected")
    
    if not 1 <= channel <= 3:
        raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
    
    try:
        success = driver.set_output(channel, state)
        
        if success:
            return {
                "status": "success",
                "channel": channel,
                "output": "ON" if state else "OFF"
            }
        else:
            raise HTTPException(status_code=500, detail="Output control failed")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== 로깅 제어 API ==========

@app.post("/logging/config")
async def configure_logging(config: LoggingConfig):
    """채널별 로깅 활성화/비활성화"""
    global logging_enabled
    
    if not 1 <= config.channel <= 3:
        raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
    
    logging_enabled[config.channel] = config.enabled
    
    return {
        "status": "success",
        "channel": config.channel,
        "table_name": f"channel_{config.channel}",
        "logging_enabled": config.enabled,
        "message": f"Channel {config.channel} logging {'enabled' if config.enabled else 'disabled'}"
    }


@app.get("/logging/config")
async def get_logging_config():
    """현재 로깅 설정 조회"""
    return {
        "logging_config": logging_enabled,
        "active_websockets": sorted(list(active_channels)),
        "tables": ["channel_1", "channel_2", "channel_3"]
    }


# ========== WebSocket (실시간 모니터링 + 로깅) ==========

@app.websocket("/ws/monitor/{channel}")
async def websocket_monitor_channel(websocket: WebSocket, channel: int):
    """채널별 독립 WebSocket - 각 채널이 독립 테이블에 저장"""
    
    if not 1 <= channel <= 3:
        await websocket.close(code=1003, reason="Invalid channel (1-3)")
        return
    
    await websocket.accept()
    active_channels.add(channel)
    print(f"WebSocket connected for channel {channel} -> table: channel_{channel}")
    print(f"Active channels: {sorted(active_channels)}")
    
    try:
        while True:
            if driver and driver.is_connected():
                try:
                    results = driver.measure_channel([channel])
                    
                    if channel in results:
                        channel_data = results[channel]
                        
                        # 로깅 활성화 여부 확인 후 채널별 테이블에 저장
                        logged = False
                        if logging_enabled.get(channel, True):
                            save_to_database(channel_data)
                            logged = True
                        
                        data = {
                            "channel": channel,
                            "table": f"channel_{channel}",
                            "voltage": channel_data.voltage,
                            "current": channel_data.current,
                            "power": channel_data.voltage * channel_data.current,
                            "timestamp": channel_data.timestamp,
                            "logged": logged
                        }
                        await websocket.send_json(data)
                    else:
                        await websocket.send_json({
                            "error": f"Channel {channel} measurement failed",
                            "logged": False
                        })
                    
                except Exception as e:
                    await websocket.send_json({
                        "error": str(e),
                        "logged": False
                    })
            else:
                await websocket.send_json({
                    "error": "Not connected",
                    "logged": False
                })
            
            # 고속 업데이트 (100ms = 10Hz)
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for channel {channel}")
    except Exception as e:
        print(f"WebSocket error for channel {channel}: {e}")
    finally:
        active_channels.discard(channel)
        print(f"Active channels: {sorted(active_channels)}")


# ========== WebSocket 상태 확인 ==========

@app.get("/ws/status")
async def get_websocket_status():
    """활성 WebSocket 연결 상태 확인"""
    return {
        "active_channels": sorted(list(active_channels)),
        "total_connections": len(active_channels),
        "connected": driver.is_connected() if driver else False,
        "logging_config": logging_enabled,
        "tables": {ch: f"channel_{ch}" for ch in [1, 2, 3]}
    }


# ========== 로그 조회 API (채널별 테이블) ==========

@app.get("/logs/recent")
async def get_recent_logs(limit: int = 100):
    """전체 채널의 최근 로그 조회 (각 채널별 테이블에서 조회)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        all_logs = []
        
        # 각 채널 테이블에서 로그 조회
        for channel in [1, 2, 3]:
            table_name = f"channel_{channel}"
            
            cursor.execute(f"""
                SELECT timestamp, datetime, voltage, current, power
                FROM {table_name}
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            for row in rows:
                all_logs.append({
                    "channel": channel,
                    "table": table_name,
                    "timestamp": row[0],
                    "datetime": row[1],
                    "voltage": row[2],
                    "current": row[3],
                    "power": row[4]
                })
        
        # 타임스탬프로 정렬
        all_logs.sort(key=lambda x: x['timestamp'], reverse=True)
        all_logs = all_logs[:limit]
        
        conn.close()
        
        return {"count": len(all_logs), "logs": all_logs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/channel/{channel}")
async def get_channel_logs(
    channel: int, 
    limit: int = 100,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """특정 채널 로그 조회 (채널별 독립 테이블)"""
    if not 1 <= channel <= 3:
        raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        table_name = f"channel_{channel}"
        
        query = f"""
            SELECT timestamp, datetime, voltage, current, power
            FROM {table_name}
            WHERE 1=1
        """
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append({
                "timestamp": row[0],
                "datetime": row[1],
                "voltage": row[2],
                "current": row[3],
                "power": row[4]
            })
        
        return {
            "channel": channel,
            "table": table_name,
            "count": len(logs),
            "logs": logs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/logs/clear")
async def clear_logs(channel: Optional[int] = None):
    """로그 삭제 (채널별 테이블 또는 전체)"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if channel is not None:
            if not 1 <= channel <= 3:
                raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
            
            table_name = f"channel_{channel}"
            cursor.execute(f"DELETE FROM {table_name}")
            message = f"Channel {channel} logs cleared (table: {table_name})"
        else:
            # 전체 채널 삭제
            for ch in [1, 2, 3]:
                cursor.execute(f"DELETE FROM channel_{ch}")
            message = "All channel logs cleared"
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return {
            "status": "success",
            "message": message,
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)