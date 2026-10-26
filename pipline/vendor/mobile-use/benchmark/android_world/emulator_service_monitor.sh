#!/bin/bash

# 服务名称（仅用于日志显示）
SERVICE_NAME="emulator"

# 启动命令
START_COMMAND="emulator -avd AndroidWorldAvd -no-window -no-audio -no-snapshot -grpc 8554"

# 检查间隔时间（秒）
INTERVAL=10

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始监控服务: $SERVICE_NAME"
echo "启动命令: $START_COMMAND"

# 函数：启动服务并记录PID
start_service() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 正在启动服务..."
    # 使用 nohup 来后台运行，并将输出重定向到文件
    nohup $START_COMMAND >> emulator_output.log 2>&1 &
    PID=$!
    echo $PID > /tmp/emulator.pid
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务已启动，PID: $PID"
}

# 检查服务是否已经在运行
check_and_init_service() {
    # 尝试通过AVD名称查找进程
    SERVICE_PID=$(pgrep -f "emulator.*AndroidWorldAvd")

    if [ -z "$SERVICE_PID" ]; then
        # 如果没有找到正在运行的服务，启动它
        start_service
    else
        # 如果找到了正在运行的服务，将其PID写入文件
        echo $SERVICE_PID > /tmp/emulator.pid
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务已在运行，PID: $SERVICE_PID 已写入到 /tmp/emulator.pid"
    fi
}

# 初始检查和可能的初始化
check_and_init_service

# 循环监控
while true; do
    sleep $INTERVAL
    
    # 检查PID文件是否存在，并读取PID值
    if [ -f /tmp/emulator.pid ]; then
        PID=$(cat /tmp/emulator.pid)
        
        # 验证PID对应的进程是否仍然存在
        if ! ps -p $PID > /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] 服务(PID: $PID)已停止，尝试重新启动服务..."
            start_service
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] PID 文件不存在，尝试初始化服务..."
        check_and_init_service
    fi
done

