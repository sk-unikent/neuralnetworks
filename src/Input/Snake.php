<?php

namespace KeltyNN\Input;

class Snake
{
    public $logs = array();
    public $tick = 0;
    public $turns = 0;
    public $score = 0;
    protected $width;
    protected $height;
    protected $snakepos = array();
    protected $snakelen = 3;
    protected $applepos;
    protected $direction = 3;

    /**
     * Initialize a game space.
     */
    public function __construct($width, $height) {
        $this->width = $width;
        $this->height = $height;

        $startx = ceil($this->width / 2);
        $starty = ceil($this->height / 2);
        for ($i = 0; $i < $this->snakelen; $i++) {
            $this->snakepos[] = array($startx + $i, $starty);
        }

        $this->spawnApple();
    }

    /**
     * Change direction.
     * 0 - up, 1 - down, 2 - left, 3 - right
     */
    public function changeDirection($direction) {
        if (($this->direction == 0 || $this->direction == 1) && ($direction == 0 || $direction == 1) && $this->direction != $direction) {
            $this->log("Refused direction {$direction}");
            return false;
        }

        if (($this->direction == 2 || $this->direction == 3) && ($direction == 2 || $direction == 3) && $this->direction != $direction) {
            $this->log("Refused direction {$direction}");
            return false;
        }

        if ($direction !== $this->direction) {
            $this->log("Changing to {$direction}");
        } else {
            $this->log("Resuming direction {$direction}");
        }

        $this->direction = $direction;

        return true;
    }

    /**
     * Returns normalised simple x/y vectors for moving to the apple.
     * x, y in. 1 means right or up, -1 means down or left
     */
    public function scoreVectors() {
        $xa = $this->applepos[0];
        $ya = $this->applepos[1];
        $current = end($this->snakepos);
        $xb = $current[0];
        $yb = $current[1];

        $this->log("Apple is at $xa, $ya");
        $this->log("Snake head is at $xb, $yb");

        return array(
            $xa == $xb ? 0 : ($xa > $xb ? 1 : -1),
            $ya == $yb ? 0 : ($ya < $yb ? 1 : -1)
        );
    }

    /**
     * Return true if the given x,y are within screen space.
     */
    protected function clamp($x, $y) {
        return !($x < 0 || $y < 0 || $x >= $this->width || $y >= $this->height);
    }

    /**
     * Move snek.
     * 0 - up, 1 - down, 2 - left, 3 - right
     */
    protected function moveSnake() {
        $current = end($this->snakepos);
        switch ($this->direction) {
            case 0:
                $current[1]--;
            break;
            case 1:
                $current[1]++;
            break;
            case 2:
                $current[0]--;
            break;
            case 3:
                $current[0]++;
            break;
        }

        if ($this->clamp($current[0], $current[1])) {
            $this->snakepos[] = $current;
            array_shift($this->snakepos);
            return true;
        }

        return false;
    }

    /**
     * Have we collided with snek?
     */
    protected function collision($check) {
        foreach ($this->snakepos as $pos) {
            if ($check[0] == $pos[0] && $check[1] == $pos[1]) {
                return true;
            }
        }

        return false;
    }

    /**
     * Spawn an apple.
     */
    public function spawnApple() {
        do {
            $this->applepos = array(rand(0, $this->width), rand(0, $this->height));
            $this->log("Spawing new apple at {$this->applepos[0]}, {$this->applepos[1]}");
        } while($this->collision($this->applepos));
    }

    /**
     * Grow the snek.
     */
    protected function growSnake() {
        $this->snakelen++;
        $current = reset($this->snakepos);
        switch ($this->direction) {
            case 0:
                $current[1]--;
            break;
            case 1:
                $current[1]++;
            break;
            case 2:
                $current[0]++;
            break;
            case 3:
                $current[0]--;
            break;
        }
        array_unshift($this->snakepos, $current);
    }

    /**
     * Submit an entry to the log.
     */
    public function log($message) {
        $this->logs[$this->tick][] = $message;
    }

    /**
     * Game tick.
     */
    public function tick() {
        $this->tick++;
        $this->logs[$this->tick] = array();

        if ($this->moveSnake()) {
            $this->turns++;
        }

        if ($this->collision($this->applepos)) {
            $this->spawnApple();
            $this->score++;

            // Grow the snek, but any more than 12 is silly. All organisms stop growing.
            if ($this->snakelen < 12) {
                $this->growSnake();
            }
        }
    }

    /**
     * What is at these coordinates?
     */
    public function whatsat($x, $y) {
        if (!$this->clamp($x, $y)) {
            return 'Bounds';
        }

        if ($this->collision(array($x, $y))) {
            return 'Snake';
        }

        if (array($x, $y) == $this->applepos) {
            return 'Apple';
        }

        return 'Nothing';
    }

    /**
     * Return the board around the snake as a normalized output.
     */
    public function exportSnakeSpace($width, $height, $exportBlank = true) {
        $current = reset($this->snakepos);
        $minx = $current[0] - ($width / 2);
        $maxx = $current[0] + ($width / 2);
        $miny = $current[1] - ($height / 2);
        $maxy = $current[1] + ($height / 2);

        $board = array();
        for ($x = $minx; $x < $maxx; $x++) {
            $board[$x] = array();
            for ($y = $miny; $y < $maxy; $y++) {
                $whatat = $this->whatsat($x, $y);
                if ($whatat == 'Apple') {
                    $board[$x][$y] = 1;
                } else if ($whatat == 'Snake') {
                    $board[$x][$y] = -1;
                } else if ($whatat == 'Bounds') {
                    $board[$x][$y] = -1;
                } if ($exportBlank) {
                    $board[$x][$y] = 0;
                }
            }
        }
        return $board;
    }

    /**
     * Return the board as a normalized output.
     */
    public function exportNormal($exportBlank = true) {
        $board = array();
        for ($x = 0; $x < $this->width; $x++) {
            $board[$x] = array();
            for ($y = 0; $y < $this->height; $y++) {
                $whatat = $this->whatsat($x, $y);
                if ($whatat == 'Apple') {
                    $board[$x][$y] = 1;
                } else if ($whatat == 'Snake') {
                    $board[$x][$y] = -1;
                } else if ($exportBlank) {
                    $board[$x][$y] = 0;
                }
            }
        }
        return $board;
    }
}
