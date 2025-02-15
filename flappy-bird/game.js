// Background scrolling speed
let move_speed = 3;
  
// Gravity constant value
let gravity = 0.5;
  
// Getting reference to the bird element
let bird = document.querySelector('.bird');
  
// Getting bird element properties
let bird_props = bird.getBoundingClientRect();
let background =
    document.querySelector('.background')
            .getBoundingClientRect();
  
// Getting reference to the score element
let score_val =
    document.querySelector('.score_val');
let message =
    document.querySelector('.message');
let score_title =
    document.querySelector('.score_title');
  
// Setting initial game state to start
let game_state = 'Start';
  
// Make bird_dy global so it's accessible everywhere
let bird_dy = 0;

// Remove the keydown event listener at the start of the file

function play() {
  function move() {
    
    // Detect if game has ended
    if (game_state != 'Play') return;
    
    // Getting reference to all the pipe elements
    let pipe_sprite = document.querySelectorAll('.pipe_sprite');
    pipe_sprite.forEach((element) => {
      
      let pipe_sprite_props = element.getBoundingClientRect();
      bird_props = bird.getBoundingClientRect();
      
      // Delete the pipes if they have moved out
      // of the screen hence saving memory
      if (pipe_sprite_props.right <= 0) {
        element.remove();
      } else {
        // Collision detection with bird and pipes
        if (
          bird_props.left < pipe_sprite_props.left +
          pipe_sprite_props.width &&
          bird_props.left +
          bird_props.width > pipe_sprite_props.left &&
          bird_props.top < pipe_sprite_props.top +
          pipe_sprite_props.height &&
          bird_props.top +
          bird_props.height > pipe_sprite_props.top
        ) {
          
          // Change game state and end the game
          // if collision occurs
          game_state = 'End';
          message.innerHTML = 'Press Enter To Restart';
          message.style.left = '28vw';
          gameOver();
          return;
        } else {
          // Increase the score if player
          // has the successfully dodged the 
          if (
            pipe_sprite_props.right < bird_props.left &&
            pipe_sprite_props.right + 
            move_speed >= bird_props.left &&
            element.increase_score == '1'
          ) {
            score_val.innerHTML = +score_val.innerHTML + 1;
          }
          element.style.left = 
            pipe_sprite_props.left - move_speed + 'px';
        }
      }
    });

    requestAnimationFrame(move);
  }
  requestAnimationFrame(move);

  function apply_gravity() {
    if (game_state != 'Play') return;
    bird_dy = bird_dy + gravity;
    
    // Remove the keyboard event listener

    if (bird_props.top <= 0 ||
        bird_props.bottom >= background.bottom) {
      game_state = 'End';
      message.innerHTML = 'Press Enter To Restart';
      message.style.left = '28vw';
      gameOver();
      return;
    }
    bird.style.top = bird_props.top + bird_dy + 'px';
    bird_props = bird.getBoundingClientRect();
    requestAnimationFrame(apply_gravity);
  }
  requestAnimationFrame(apply_gravity);

  let pipe_seperation = 0;
  
  // Constant value for the gap between two pipes
  let pipe_gap = 35;
  function create_pipe() {
    if (game_state != 'Play') return;
    
    // Create another set of pipes
    // if distance between two pipe has exceeded
    // a predefined value
    if (pipe_seperation > 115) {
      pipe_seperation = 0
      
      // Calculate random position of pipes on y axis
      let pipe_posi = Math.floor(Math.random() * 43) + 8;
      let pipe_sprite_inv = document.createElement('div');
      pipe_sprite_inv.className = 'pipe_sprite';
      pipe_sprite_inv.style.top = pipe_posi - 70 + 'vh';
      pipe_sprite_inv.style.left = '100vw';
      
      // Append the created pipe element in DOM
      document.body.appendChild(pipe_sprite_inv);
      let pipe_sprite = document.createElement('div');
      pipe_sprite.className = 'pipe_sprite';
      pipe_sprite.style.top = pipe_posi + pipe_gap + 'vh';
      pipe_sprite.style.left = '100vw';
      pipe_sprite.increase_score = '1';
      
      // Append the created pipe element in DOM
      document.body.appendChild(pipe_sprite);
    }
    pipe_seperation++;
    requestAnimationFrame(create_pipe);
  }
  requestAnimationFrame(create_pipe);
}

// Replace the NeuralNetwork class with these functions
async function predictMove(gameState) {
    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ...gameState,
                windowWidth: window.innerWidth,
                windowHeight: window.innerHeight
            })
        });
        const data = await response.json();
        return data.should_jump;
    } catch (error) {
        console.error('AI prediction failed:', error);
        return false;
    }
}

async function trainAI(trainingData) {
    try {
        await fetch('http://localhost:5000/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(trainingData)
        });
    } catch (error) {
        console.error('AI training failed:', error);
    }
}

const ai = new NeuralNetwork();
let trainingData = [];

// Modify the aiControl function
async function aiControl() {
    if (game_state === 'Play') {
        const gameState = {
            birdY: bird_props.top,
            birdVelocity: bird_dy,
            nearestPipe: getNearestPipe()
        };
        
        const shouldJump = await predictMove(gameState);
        if (shouldJump) {
            bird_dy = -7.6;
            trainingData.push({
                state: gameState,
                action: 1
            });
        } else {
            trainingData.push({
                state: gameState,
                action: 0
            });
        }
    }
}

// Call AI control every frame
setInterval(aiControl, 100);

// When game ends, train the network
function gameOver() {
    if (trainingData.length > 0) {
        trainAI(trainingData);
        trainingData = [];
    }
    
    // Auto restart without message
    setTimeout(() => {
        document.querySelectorAll('.pipe_sprite').forEach((e) => {
            e.remove();
        });
        bird.style.top = '40vh';
        game_state = 'Play';
        message.innerHTML = '';
        score_title.innerHTML = 'Score : ';
        score_val.innerHTML = '0';
        play();
    }, 1000);
}

// Add helper function to get nearest pipe
function getNearestPipe() {
    const pipes = document.querySelectorAll('.pipe_sprite');
    let nearestPipe = null;
    let minDist = Infinity;
    
    pipes.forEach(pipe => {
        const pipeProps = pipe.getBoundingClientRect();
        const dist = pipeProps.left - bird_props.left;
        if (dist > 0 && dist < minDist) {
            minDist = dist;
            nearestPipe = pipeProps;
        }
    });
    return nearestPipe;
}

// Start the game automatically
window.onload = function() {
    game_state = 'Play';
    message.innerHTML = '';
    score_title.innerHTML = 'Score : ';
    score_val.innerHTML = '0';
    play();
    setInterval(aiControl, 50); // Made interval shorter for better responsiveness
};
