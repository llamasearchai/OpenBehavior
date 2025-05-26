#!/bin/bash

# OpenBehavior Docker Start Script

set -e

echo "Starting OpenBehavior platform..."

# Set default environment variables
export PYTHONPATH="${PYTHONPATH:-/app/python}"
export OPENBEHAVIOR_CONFIG_PATH="${OPENBEHAVIOR_CONFIG_PATH:-/app/config/production.yaml}"

# Function to wait for service
wait_for_service() {
    local host=$1
    local port=$2
    local service=$3
    
    echo "Waiting for $service to be ready..."
    while ! nc -z $host $port; do
        sleep 1
    done
    echo "$service is ready!"
}

# Wait for dependencies if they exist
if [ ! -z "$REDIS_URL" ]; then
    REDIS_HOST=$(echo $REDIS_URL | cut -d'/' -f3 | cut -d':' -f1)
    REDIS_PORT=$(echo $REDIS_URL | cut -d'/' -f3 | cut -d':' -f2)
    wait_for_service $REDIS_HOST $REDIS_PORT "Redis"
fi

if [ ! -z "$MONGODB_URL" ]; then
    MONGO_HOST=$(echo $MONGODB_URL | cut -d'/' -f3 | cut -d':' -f1)
    MONGO_PORT=$(echo $MONGODB_URL | cut -d'/' -f3 | cut -d':' -f2)
    wait_for_service $MONGO_HOST $MONGO_PORT "MongoDB"
fi

# Initialize database if needed
if [ "$OPENBEHAVIOR_ENV" = "production" ]; then
    echo "Initializing production database..."
    python -c "
import asyncio
from openbehavior.utils.database import DatabaseManager

async def init_db():
    db = DatabaseManager()
    await db.initialize()
    await db.close()

asyncio.run(init_db())
" || echo "Database initialization skipped or failed"
fi

# Start the application based on the mode
if [ "$1" = "api" ]; then
    echo "Starting OpenBehavior API server..."
    exec python -m openbehavior.api.main
elif [ "$1" = "cli" ]; then
    echo "Starting OpenBehavior CLI..."
    exec python -m openbehavior.cli.main "${@:2}"
elif [ "$1" = "worker" ]; then
    echo "Starting OpenBehavior worker..."
    exec python -m openbehavior.worker.main
else
    # Default: start both API and dashboard
    echo "Starting OpenBehavior API and Dashboard..."
    
    # Start API server in background
    python -m openbehavior.api.main &
    API_PID=$!
    
    # Start dashboard if it exists
    if [ -d "/app/dashboard" ]; then
        echo "Starting dashboard..."
        cd /app/dashboard && python -m http.server 3000 &
        DASHBOARD_PID=$!
    fi
    
    # Function to handle shutdown
    shutdown() {
        echo "Shutting down OpenBehavior..."
        kill $API_PID 2>/dev/null || true
        [ ! -z "$DASHBOARD_PID" ] && kill $DASHBOARD_PID 2>/dev/null || true
        exit 0
    }
    
    # Set up signal handlers
    trap shutdown SIGTERM SIGINT
    
    # Wait for processes
    wait $API_PID
fi 