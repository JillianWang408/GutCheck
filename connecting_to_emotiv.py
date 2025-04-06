#!/usr/bin/env python3
# Python WebSocket Client for Cortex API
# First install the 'websocket-client' package: pip install websocket-client
import json
import ssl
import websocket
# Define callback functions
def on_open(ws):
    print("Connection opened")
    # Send the getCortexInfo request
    message = {
        "id": 1,
        "jsonrpc": "2.0",
        "method": "getCortexInfo"
    }
    ws.send(json.dumps(message))
    print(f"Sent request: {message}")
def on_message(ws, message):
    print(f"Received raw response: {message}")
    # Parse and display the response in a more readable format
    try:
        response = json.loads(message)
        print(f"Parsed response: {json.dumps(response, indent=2)}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
def on_error(ws, error):
    print(f"WebSocket error: {error}")
def on_close(ws, close_status_code, close_msg):
    print(f"Connection closed. Code: {close_status_code}, Message: {close_msg}")
# Create and start the WebSocket connection
if __name__ == "__main__":
    # Disable SSL certificate verification if needed (for self-signed certificates)
    websocket.enableTrace(True)  # Enable for verbose logging
    ws = websocket.WebSocketApp("wss://localhost:6868",
                               on_open=on_open,
                               on_message=on_message,
                               on_error=on_error,
                               on_close=on_close)
    # If using a self-signed certificate, use this:
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    # If using a valid certificate, use this instead:
    # ws.run_forever()