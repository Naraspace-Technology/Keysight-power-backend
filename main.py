"""
Keysight E36313A Power Supply Control Backend
FastAPI 기반 REST API 서버
"""
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

from keysight_driver import KeysightE36313ADriver, ChannelData

# ========== 전역 변수 ==========
driver: Optional[KeysightE36313ADriver] = None
import os
db_path = Path(os.getenv("DATABASE_PATH", "power_supply_logs.db"))

# 활성 WebSocket 연결 추적
active_channels: Set[int] = set()


# ========== Database 초기화 ==========
def init_database():
    """SQLite 데이터베이스 초기화"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS measurement_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            datetime TEXT NOT NULL,
            channel INTEGER NOT NULL,
            voltage REAL NOT NULL,
            current REAL NOT NULL,
            power REAL NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp 
        ON measurement_logs(timestamp)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_channel 
        ON measurement_logs(channel)
    """)
    
    conn.commit()
    conn.close()


def save_to_database(data: ChannelData):
    """측정 데이터 저장"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO measurement_logs 
            (timestamp, datetime, channel, voltage, current, power)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            data.timestamp,
            datetime.fromtimestamp(data.timestamp).isoformat(),
            data.channel,
            data.voltage,
            data.current,
            data.voltage * data.current
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Database save error: {e}")


# ========== Lifespan 관리 ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    # Startup
    init_database()
    print("Database initialized")
    yield
    # Shutdown
    if driver and driver.is_connected():
        driver.disconnect()
    print("Application shutdown")


# ========== FastAPI 앱 초기화 ==========
app = FastAPI(
    title="Keysight E36313A Control API",
    description="Power Supply Control Backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 origin으로 제한
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


class AllChannelsSetting(BaseModel):
    voltages: List[float] = Field(..., min_items=3, max_items=3)
    currents: List[float] = Field(..., min_items=3, max_items=3)


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
        "status": "running"
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
        data = driver.measure_channel(channel)
        if data is None:
            raise HTTPException(status_code=500, detail="Measurement failed")
        
        # 데이터베이스 저장
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


# @app.get("/measure/all")
# async def measure_all_channels():
#     """전체 채널 일괄 측정"""
#     if not driver or not driver.is_connected():
#         raise HTTPException(status_code=400, detail="Not connected")
#     
#     try:
#         results: Dict[int, ChannelData] = driver.measure_all_channels()
#         
#         # 데이터베이스 저장
#        for data in results.values():
#             save_to_database(data)
#         
#         response = {"channels": []}
#         for ch, data in results.items():
#             response["channels"].append({
#                 "channel": data.channel,
#                 "voltage": data.voltage,
#                 "current": data.current,
#                 "power": round(data.voltage * data.current, 6),  # 소수점 안정성
#                 "timestamp": data.timestamp,
#                 "datetime": datetime.fromtimestamp(data.timestamp).isoformat()
#             })
#         
#         return response
#         
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Measurement error: {str(e)}")


# ========== 설정 API ==========

@app.post("/set/volt/{channel}")
async def set_voltage(channel: int, setting: voltageSetting):
    """개별 채널 전압 설정"""
    if not driver or not driver.is_connected():
        raise HTTPException(status_code=400, detail="Not connected")
    
    if not 1 <= channel <= 3:
        raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
    
    try:
        # 범위 체크
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
        # 범위 체크
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


# @app.post("/set/volt/all")
# async def set_all_channels(setting: AllChannelsSetting):
#     """전체 채널 일괄 설정"""
#     if not driver or not driver.is_connected():
#         raise HTTPException(status_code=400, detail="Not connected")
#     
#     try:
#         # 범위 체크
#         for ch in range(1, 4):
#             if setting.voltages[ch-1] > driver.MAX_VOLTAGE[ch]:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"CH{ch} voltage exceeds max ({driver.MAX_VOLTAGE[ch]}V)"
#                 )
#         
#         results = driver.set_all_channels(setting.voltages, setting.currents)
#         
#         return {
#             "status": "success" if all(results.values()) else "partial",
#             "results": results,
#             "settings": {
#                 "voltages": setting.voltages
#             }
#         }
#         
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/set/current/all")
# async def set_all_channels_current(setting: AllChannelsSetting):
#     """전체 채널 일괄 전류 설정"""
#     if not driver or not driver.is_connected():
#         raise HTTPException(status_code=400, detail="Not connected")
#     
#     try:
#         # 범위 체크
#         for ch in range(1, 4):
#             if setting.currents[ch-1] > driver.MAX_CURRENT[ch]:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"CH{ch} current exceeds max ({driver.MAX_CURRENT[ch]}A)"
#                 )
#         
#         results = driver.set_all_channels_current(setting.currents)
#         
#         return {
#             "status": "success" if all(results.values()) else "partial",
#             "results": results,
#             "settings": {
#                 "currents": setting.currents
#             }
#         }
#         
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

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


# @app.post("/output/all/{state}")
# async def set_all_outputs(state: bool):
#     """전체 채널 출력 일괄 ON/OFF"""
#     if not driver or not driver.is_connected():
#         raise HTTPException(status_code=400, detail="Not connected")
#     
#     try:
#         results = driver.set_all_outputs(state)
#         
#         return {
#             "status": "success" if all(results.values()) else "partial",
#             "output": "ON" if state else "OFF",
#             "results": results
#         }
#         
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# ========== WebSocket (실시간 모니터링) ==========
@app.websocket("/ws/monitor/{channel}")
async def websocket_monitor_channel(websocket: WebSocket, channel: int):
    """채널별 독립 WebSocket - 각 채널이 독립적으로 측정"""
    
    # 채널 번호 검증
    if not 1 <= channel <= 3:
        await websocket.close(code=1003, reason="Invalid channel (1-3)")
        return
    
    await websocket.accept()
    active_channels.add(channel)
    print(f"✓ WebSocket connected for channel {channel}")
    print(f"  Active channels: {sorted(active_channels)}")
    
    try:
        while True:
            if driver and driver.is_connected():
                try:
                    # 단일 채널만 측정 (독립적으로 빠르게)
                    results = driver.measure_channel([channel])
                    
                    if channel in results:
                        channel_data = results[channel]
                        data = {
                            "channel": channel,
                            "voltage": channel_data.voltage,
                            "current": channel_data.current,
                            "power": channel_data.voltage * channel_data.current,
                            "timestamp": channel_data.timestamp
                        }
                        await websocket.send_json(data)
                    else:
                        await websocket.send_json({
                            "error": f"Channel {channel} measurement failed"
                        })
                    
                except Exception as e:
                    await websocket.send_json({"error": str(e)})
            else:
                await websocket.send_json({"error": "Not connected"})
            
            # 고속 업데이트 (100ms = 10Hz)
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print(f"✗ WebSocket disconnected for channel {channel}")
    except Exception as e:
        print(f"✗ WebSocket error for channel {channel}: {e}")
    finally:
        active_channels.discard(channel)
        print(f"  Active channels: {sorted(active_channels)}")


# ========== 활성 연결 상태 확인 API ==========

@app.get("/ws/status")
async def get_websocket_status():
    """활성 WebSocket 연결 상태 확인"""
    return {
        "active_channels": sorted(list(active_channels)),
        "total_connections": len(active_channels),
        "connected": driver.is_connected() if driver else False
    }

# ========== 데이터 로그 조회 API ==========

@app.get("/logs/recent")
async def get_recent_logs(limit: int = 100):
    """최근 로그 조회"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, datetime, channel, voltage, current, power
            FROM measurement_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        logs = []
        for row in rows:
            logs.append({
                "timestamp": row[0],
                "datetime": row[1],
                "channel": row[2],
                "voltage": row[3],
                "current": row[4],
                "power": row[5]
            })
        
        return {"count": len(logs), "logs": logs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs/{channel}")
async def get_channel_logs(
    channel: int, 
    limit: int = 100,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None
):
    """특정 채널 로그 조회"""
    if not 1 <= channel <= 3:
        raise HTTPException(status_code=400, detail="Invalid channel (1-3)")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        query = """
            SELECT timestamp, datetime, voltage, current, power
            FROM measurement_logs
            WHERE channel = ?
        """
        params = [channel]
        
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
        
        return {"channel": channel, "count": len(logs), "logs": logs}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)