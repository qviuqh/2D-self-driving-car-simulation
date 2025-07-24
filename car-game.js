// Game initialization
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const carImage = new Image();
const mapImage = new Image();
let car;
let mapCanvas;
let mapCtx;

let aiEnabled = false;
let neuralNet = null;

// Game variables
let gameInitialized = false;

// Neural network visualization variables
let lastActivations = { input: [], hidden: [], output: [] };
const networkStructure = {
    inputSize: 6,
    hiddenSize: 8,
    outputSize: 4
};

// Simple Neural Network for AI
class SimpleNeuralNetwork {
    constructor(weights) {
        this.inputSize = 6;
        this.hiddenSize = 8;
        this.outputSize = 4;
        this.setWeights(weights);
    }

    setWeights(weights) {
        let idx = 0;
        this.w1 = Array.from({ length: 6 }, () =>
            Array.from({ length: 8 }, () => weights[idx++])
        );
        this.b1 = weights.slice(idx, idx + 8);
        idx += 8;
        this.w2 = Array.from({ length: 8 }, () =>
            Array.from({ length: 4 }, () => weights[idx++])
        );
        this.b2 = weights.slice(idx, idx + 4);
    }

    forward(input) {
        const tanh = x => Math.tanh(x);
        const sigmoid = x => 1 / (1 + Math.exp(-x));
        
        // Store input activations for visualization
        lastActivations.input = [...input];
        
        // Hidden layer
        const h = this.b1.map((b, j) =>
            tanh(input.reduce((sum, x, i) => sum + x * this.w1[i][j], b))
        );
        lastActivations.hidden = [...h];
        
        // Output layer
        const output = this.b2.map((b, k) =>
            sigmoid(h.reduce((sum, h_j, j) => sum + h_j * this.w2[j][k], b))
        );
        lastActivations.output = [...output];
        
        return output;
    }

    getAction(state) {
        const [up, down, left, right] = this.forward(state);
        const steering = right - left;
        const throttle = up - down;
        return [
            Math.max(-1, Math.min(1, steering)),
            Math.max(-1, Math.min(1, throttle))
        ];
    }
}

// Camera
const camera = {
    x: 0,
    y: 0,
    width: canvas.width,
    height: canvas.height
};

// Control states
const keys = {
    up: false,
    down: false,
    left: false,
    right: false
};

// Sensors
const sensors = {
    front: { angle: 0, distance: 300, detection: Infinity },
    left: { angle: -40 * Math.PI / 180, distance: 250, detection: Infinity },
    right: { angle: 40 * Math.PI / 180, distance: 250, detection: Infinity },
    left_1: { angle: -20 * Math.PI / 180, distance: 250, detection: Infinity },
    right_2: { angle: 20 * Math.PI / 180, distance: 250, detection: Infinity }
};

// Initialize game when map image loads
mapImage.onload = function() {
    car = {
        x: 600,
        y: 1650,
        width: 50,
        height: 30,
        radius: 15,
        angle: 0,
        velocity: { x: 0, y: 0 },
        speed: 0,
        maxSpeed: 10,
        acceleration: 0,
        maxAcceleration: 0.5,
        friction: 0.96,
        turnSpeed: 0.065,
        color: '#3498db'
    };
    
    mapCanvas = document.createElement('canvas');
    mapCanvas.width = mapImage.width;
    mapCanvas.height = mapImage.height;
    mapCtx = mapCanvas.getContext('2d', { willReadFrequently: true });
    mapCtx.drawImage(mapImage, 0, 0);

    camera.x = car.x - camera.width / 2;
    camera.y = car.y - camera.height / 2;
    
    gameInitialized = true;
    
    // Initialize neural network visualization
    drawNeuralNetwork();
    
    gameLoop();
};

// Load images (using placeholder gray rectangles if images not available)
carImage.onerror = function() {
    const canvas = document.createElement('canvas');
    canvas.width = 50;
    canvas.height = 30;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#3498db';
    ctx.fillRect(0, 0, 50, 30);
    ctx.fillStyle = '#2980b9';
    ctx.fillRect(10, 5, 30, 20);
    carImage.src = canvas.toDataURL();
};

mapImage.onerror = function() {
    const canvas = document.createElement('canvas');
    canvas.width = 1000;
    canvas.height = 800;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#27ae60';
    ctx.fillRect(0, 0, 1000, 800);
    ctx.fillStyle = '#5e5e5e';
    ctx.fillRect(100, 200, 800, 400);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 4;
    ctx.setLineDash([20, 10]);
    ctx.beginPath();
    ctx.moveTo(100, 400);
    ctx.lineTo(900, 400);
    ctx.stroke();
    mapImage.src = canvas.toDataURL();
};

carImage.src = 'static/car_blue_3.png';
mapImage.src = 'static/track_map.png';

// Event listeners for keyboard
document.addEventListener('keydown', (e) => {
    switch(e.key.toLowerCase()) {
        case 'w':
        case 'arrowup':
            keys.up = true;
            e.preventDefault();
            break;
        case 's':
        case 'arrowdown':
            keys.down = true;
            e.preventDefault();
            break;
        case 'a':
        case 'arrowleft':
            keys.left = true;
            e.preventDefault();
            break;
        case 'd':
        case 'arrowright':
            keys.right = true;
            e.preventDefault();
            break;
    }
});

document.addEventListener('keyup', (e) => {
    switch(e.key.toLowerCase()) {
        case 'w':
        case 'arrowup':
            keys.up = false;
            break;
        case 's':
        case 'arrowdown':
            keys.down = false;
            break;
        case 'a':
        case 'arrowleft':
            keys.left = false;
            break;
        case 'd':
        case 'arrowright':
            keys.right = false;
            break;
    }
});

// Game utility functions
function isOnRoad(x, y) {
    if (!mapCtx) return true;
    
    const pixel = mapCtx.getImageData(x, y, 1, 1).data;
    const [r, g, b] = pixel;

    return (
        Math.abs(r - 94) <= 10 &&
        Math.abs(g - 94) <= 10 &&
        Math.abs(b - 94) <= 10
    );
}

function isWhiteLine(x, y) {
    if (!mapCtx) return false;
    
    x = Math.max(0, Math.min(Math.floor(x), mapImage.width - 1));
    y = Math.max(0, Math.min(Math.floor(y), mapImage.height - 1));
    const pixel = mapCtx.getImageData(x, y, 1, 1).data;
    return pixel[0] > 220 && pixel[1] > 220 && pixel[2] > 220;
}

function updateCamera() {
    if (!car) return;  // Thêm điều kiện kiểm tra
    
    camera.x = car.x - camera.width / 2;
    camera.y = car.y - camera.height / 2;

    camera.x = Math.max(0, Math.min(camera.x, mapImage.width - camera.width));
    camera.y = Math.max(0, Math.min(camera.y, mapImage.height - camera.height));
}

function updateSensors() {
    if (!gameInitialized) return;
    
    Object.keys(sensors).forEach(sensorKey => {
        const sensor = sensors[sensorKey];
        const sensorAngle = car.angle + sensor.angle;

        sensor.detection = sensor.distance;

        for (let d = 10; d <= sensor.distance; d += 5) {
            const px = car.x + Math.cos(sensorAngle) * d;
            const py = car.y + Math.sin(sensorAngle) * d;
            
            if (px < 0 || px >= mapImage.width || py < 0 || py >= mapImage.height) {
                sensor.detection = d - 5;
                break;
            }
            
            if (isWhiteLine(px, py)) {
                sensor.detection = d;
                break;
            }
            
            if (!isOnRoad(px, py)) {
                sensor.detection = Math.max(10, d - 5);
                break;
            }
        }
    });
}

function updateCarPhysics() {
    if (!gameInitialized) return;

    const maxA = car.maxAcceleration;
    const v = Math.abs(car.speed);
    const vmax = car.maxSpeed;
    function customAcceleration(v) {
        return maxA * (1 - v / vmax);
    }

    let up = keys.up;
    let down = keys.down;
    let left = keys.left;
    let right = keys.right;

    // AI override controls
    if (aiEnabled && neuralNet) {
        const state = [
            Math.abs(car.speed) / car.maxSpeed,
            (sensors.front.detection === Infinity ? 300 : sensors.front.detection) / 300,
            (sensors.left.detection === Infinity ? 250 : sensors.left.detection) / 250,
            (sensors.right.detection === Infinity ? 250 : sensors.right.detection) / 250,
            (sensors.left_1.detection === Infinity ? 250 : sensors.left_1.detection) / 250,
            (sensors.right_2.detection === Infinity ? 250 : sensors.right_2.detection) / 250
        ];

        const [steering, throttle] = neuralNet.getAction(state);

        up = throttle > 0.1;
        down = throttle < -0.1;
        left = steering < -0.1;
        right = steering > 0.1;
    }

    if (up) {
        car.acceleration += (customAcceleration(v) - car.acceleration) * 0.2;
    } else if (down) {
        car.acceleration += ((-customAcceleration(v) * 0.5) - car.acceleration) * 0.2;
    } else {
        car.acceleration += (0 - car.acceleration) * 0.02;
    }

    car.speed += car.acceleration;
    car.speed *= car.friction;
    car.speed = Math.min(car.maxSpeed, car.speed);

    if (Math.abs(car.speed) > 0.1) {
        if (left) {
            car.angle -= car.turnSpeed * Math.abs(car.speed) / car.maxSpeed;
        }
        if (right) {
            car.angle += car.turnSpeed * Math.abs(car.speed) / car.maxSpeed;
        }
    }

    car.velocity.x = Math.cos(car.angle) * car.speed;
    car.velocity.y = Math.sin(car.angle) * car.speed;

    car.x += car.velocity.x;
    car.y += car.velocity.y;

    if (car.x < car.radius) {
        car.x = car.radius;
        car.velocity.x *= -0.5;
        car.speed *= 0.5;
    }
    if (car.x > mapImage.width - car.radius) {
        car.x = mapImage.width - car.radius;
        car.velocity.x *= -0.5;
        car.speed *= 0.5;
    }
    if (car.y < car.radius) {
        car.y = car.radius;
        car.velocity.y *= -0.5;
        car.speed *= 0.5;
    }
    if (car.y > mapImage.height - car.radius) {
        car.y = mapImage.height - car.radius;
        car.velocity.y *= -0.5;
        car.speed *= 0.5;
    }

    if (!isOnRoad(car.x, car.y)) {
        car.x -= car.velocity.x * 0.5;
        car.y -= car.velocity.y * 0.5;
        car.speed *= -0.3;
    }
}

// Drawing functions
function drawBackground() {
    if (!gameInitialized) return;
    
    ctx.drawImage(
        mapImage,
        camera.x, camera.y, camera.width, camera.height,
        0, 0, canvas.width, canvas.height
    );
}

function drawCar() {
    if (!gameInitialized) return;
    
    ctx.save();
    ctx.translate(car.x - camera.x, car.y - camera.y);
    ctx.rotate(car.angle);
    ctx.drawImage(carImage, -car.width / 2, -car.height / 2, car.width, car.height);
    ctx.restore();
}

function drawSensors() {
    if (!gameInitialized) return;
    
    Object.keys(sensors).forEach(sensorKey => {
        const sensor = sensors[sensorKey];
        const sensorAngle = car.angle + sensor.angle;
        
        ctx.strokeStyle = sensor.detection < sensor.distance ? '#e74c3c' : '#8c8c8c';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(car.x - camera.x, car.y - camera.y);
        ctx.lineTo(
            car.x - camera.x + Math.cos(sensorAngle) * sensor.detection,
            car.y - camera.y + Math.sin(sensorAngle) * sensor.detection
        );
        ctx.stroke();
        
        if (sensor.detection < sensor.distance) {
            ctx.fillStyle = '#e74c3c';
            ctx.beginPath();
            ctx.arc(
                car.x - camera.x + Math.cos(sensorAngle) * sensor.detection,
                car.y - camera.y + Math.sin(sensorAngle) * sensor.detection,
                4, 0, Math.PI * 2
            );
            ctx.fill();
        }
    });
}

function updateUI() {
    if (!gameInitialized) return;
    
    document.getElementById('speed').textContent = `${Math.abs(car.speed * 10).toFixed(1)} km/h`;
    document.getElementById('angle').textContent = `${(car.angle * 180 / Math.PI).toFixed(1)}°`;
    document.getElementById('acceleration').textContent = `${(car.acceleration * 20).toFixed(1)} m/s²`;
    document.getElementById('position').textContent = `${car.x.toFixed(0)}, ${car.y.toFixed(0)}`;
    
    // Update neural network visualization
    if (aiEnabled && neuralNet) {
        drawNeuralNetwork();
    }
}

// Neural Network Visualization Functions
function drawNeuralNetwork() {
    const svg = document.getElementById('networkSvg');
    if (!svg) return;
    
    const svgRect = svg.getBoundingClientRect();
    const width = svgRect.width || 300;
    const height = 350;
    
    svg.innerHTML = '';
    
    const layerX = [50, width/2, width-50];
    const layerSizes = [networkStructure.inputSize, networkStructure.hiddenSize, networkStructure.outputSize];
    const layerLabels = [
        ['Speed', 'Front', 'Left', 'Right', 'Left-1', 'Right-1'], 
        Array(8).fill(' '), 
        ['Up', 'Down', 'Left', 'Right']
    ];
    
    // Track which nodes have active connections
    const activeNodes = {
        input: new Set(),
        hidden: new Set(),
        output: new Set()
    };

    // Thêm constant cho ngưỡng kích hoạt
    const ACTIVATION_THRESHOLD = 0.2;  // Có thể điều chỉnh giá trị này
    const CONNECTION_THRESHOLD = 0.1;  // Ngưỡng cho connections

    // Draw connections first and track active nodes
    for (let layer = 0; layer < 2; layer++) {
        for (let i = 0; i < layerSizes[layer]; i++) {
            for (let j = 0; j < layerSizes[layer + 1]; j++) {
                const y1 = 50 + (i + 0.5) * (height - 100) / layerSizes[layer];
                const y2 = 50 + (j + 0.5) * (height - 100) / layerSizes[layer + 1];
                
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('x1', layerX[layer]);
                line.setAttribute('y1', y1);
                line.setAttribute('x2', layerX[layer + 1]);
                line.setAttribute('y2', y2);
                line.setAttribute('class', 'connection');
                
                // Check connection activation với ngưỡng mới
                if (aiEnabled && lastActivations.input.length > 0) {
                    let weight = 0;
                    let activation = 0;
                    
                    if (layer === 0) {
                        weight = neuralNet.w1[i][j];
                        activation = lastActivations.input[i] * lastActivations.hidden[j];
                    } else {
                        weight = neuralNet.w2[i][j];
                        activation = lastActivations.hidden[i] * lastActivations.output[j];
                    }

                    // Sử dụng CONNECTION_THRESHOLD
                    if (Math.abs(activation) > CONNECTION_THRESHOLD) {
                        if (layer === 0) {
                            activeNodes.input.add(i);
                            activeNodes.hidden.add(j);
                        } else {
                            activeNodes.hidden.add(i);
                            activeNodes.output.add(j);
                        }
                        line.setAttribute('class', `connection ${weight >= 0 ? 'active-positive' : 'active-negative'}`);
                    }
                }
                
                svg.appendChild(line);
            }
        }
    }
    
    // Draw neurons with threshold
    for (let layer = 0; layer < 3; layer++) {
        for (let i = 0; i < layerSizes[layer]; i++) {
            const y = 50 + (i + 0.5) * (height - 100) / layerSizes[layer];
            
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', layerX[layer]);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', 12);
            circle.setAttribute('class', 'neuron');
            
            if (aiEnabled && lastActivations.input.length > 0) {
                let activation = 0;
                
                // Get raw activation value
                if (layer === 0) {
                    activation = lastActivations.input[i];
                } else if (layer === 1) {
                    activation = lastActivations.hidden[i];
                } else {
                    activation = lastActivations.output[i];
                }
                
                // Check both node active state và activation threshold
                let isActive = false;
                if (layer === 0) {
                    isActive = activeNodes.input.has(i) && Math.abs(activation) > ACTIVATION_THRESHOLD;
                } else if (layer === 1) {
                    isActive = activeNodes.hidden.has(i) && Math.abs(activation) > ACTIVATION_THRESHOLD;
                } else {
                    isActive = activeNodes.output.has(i) && Math.abs(activation) > ACTIVATION_THRESHOLD;
                }
                
                if (isActive) {
                    circle.setAttribute('class', 'neuron active');
                    const intensity = Math.max(0, Math.min(1, (Math.abs(activation) + 1) / 2));
                    const color = interpolateColor('#666', '#ffd700', intensity);
                    circle.style.fill = color;
                }
            }
            
            svg.appendChild(circle);
            
            // Add labels
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', layerX[layer]);
            text.setAttribute('y', y - 18);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('fill', 'white');
            text.setAttribute('font-size', '10');
            text.textContent = layerLabels[layer][i];
            svg.appendChild(text);
        }
    }
}

// Add helper function for color interpolation
function interpolateColor(color1, color2, factor) {
    const r1 = parseInt(color1.slice(1, 3), 16);
    const g1 = parseInt(color1.slice(3, 5), 16);
    const b1 = parseInt(color1.slice(5, 7), 16);
    
    const r2 = parseInt(color2.slice(1, 3), 16);
    const g2 = parseInt(color2.slice(3, 5), 16);
    const b2 = parseInt(color2.slice(5, 7), 16);
    
    const r = Math.round(r1 + (r2 - r1) * factor);
    const g = Math.round(g1 + (g2 - g1) * factor);
    const b = Math.round(b1 + (b2 - b1) * factor);
    
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

// Public functions (called from HTML)
function resetCar() {
    if (!gameInitialized) return;
    
    car.x = 600;
    car.y = 1650;
    car.angle = 0;
    car.velocity = { x: 0, y: 0 };
    car.speed = 0;
    car.acceleration = 0;
    
    // Reset activations
    lastActivations = { input: [], hidden: [], output: [] };
    drawNeuralNetwork();
}

// Enable AI driving
async function enableAI() {
    try {
        const res = await fetch("genetic_multilap_gen_30.json");
        const data = await res.json();
        neuralNet = new SimpleNeuralNetwork(data.weights);
        aiEnabled = true;
        console.log("✅ AI đã bật");
        
        // Update button styling
        const buttons = document.querySelectorAll('.control-btn');
        buttons.forEach(btn => {
            if (btn.textContent.includes('Bật AI')) {
                btn.style.background = 'linear-gradient(45deg, #27ae60, #2ecc71)';
                btn.style.boxShadow = '0 5px 15px rgba(39, 174, 96, 0.4)';
            }
        });
        
        drawNeuralNetwork();
    } catch (error) {
        console.warn("Không thể tải file AI weights, sử dụng weights mặc định");
        // Fallback weights (sample)
        const fallbackWeights = Array(92).fill(0).map(() => (Math.random() - 0.5) * 2);
        neuralNet = new SimpleNeuralNetwork(fallbackWeights);
        aiEnabled = true;
        drawNeuralNetwork();
    }
}

// Disable AI driving
function disableAI() {
    neuralNet = null;
    aiEnabled = false;
    console.log("🚫 AI đã tắt");
    
    // Update button styling
    const buttons = document.querySelectorAll('.control-btn');
    buttons.forEach(btn => {
        if (btn.textContent.includes('Bật AI')) {
            btn.style.background = 'linear-gradient(45deg, #ff6b6b, #ee5a24)';
            btn.style.boxShadow = '0 5px 15px rgba(255, 107, 107, 0.4)';
        }
    });
    
    // Reset activations and redraw network
    lastActivations = { input: [], hidden: [], output: [] };
    drawNeuralNetwork();
}

// Main game loop
function gameLoop() {
    if (!gameInitialized || !car) {  // Thêm điều kiện kiểm tra car
        requestAnimationFrame(gameLoop);
        return;
    }
    
    updateCarPhysics();
    updateCamera();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    drawBackground();
    updateSensors();
    drawSensors();
    drawCar();
    updateUI();

    requestAnimationFrame(gameLoop);
}

// Initialize neural network visualization on page load
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for the page to fully load
    setTimeout(() => {
        drawNeuralNetwork();
    }, 100);
});