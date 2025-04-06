// Browser-based WebSocket Client for Cortex API
// You can run this code in your browser's console
// Create a WebSocket connection
const socket = new WebSocket('wss://localhost:6868');
// Connection opened
socket.addEventListener('open', (event) => {
    console.log('Connected to Cortex API');
    // Send the getCortexInfo request
    const message = {
        id: 1,
        jsonrpc: "2.0",
        method: "getCortexInfo"
    };
    socket.send(JSON.stringify(message));
    console.log('Sent request:', message);
});
// Listen for messages
socket.addEventListener('message', (event) => {
    console.log('Received response:', event.data);
    // Parse and display the response in a more readable format
    try {
        const response = JSON.parse(event.data);
        console.log('Parsed response:', response);
    } catch (error) {
        console.error('Failed to parse response:', error);
    }
});
// Listen for connection errors
socket.addEventListener('error', (event) => {
    console.error('WebSocket error:', event);
});
// Connection closed
socket.addEventListener('close', (event) => {
    console.log('Connection closed. Code:', event.code, 'Reason:', event.reason);
});