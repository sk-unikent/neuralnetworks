<?php
require_once(dirname(__FILE__) . '/../lib.php');

$network = new KeltyNN\Networks\FFMLFlexPerceptron(16, 0, 4);
$network->setDesignerMode();
$trainer = new KeltyNN\Trainers\Genetic($network);

// Train a network for this game.
$trainer->run(function($ontick) {
    $gamespace = new \KeltyNN\Input\Snake(50, 50);
    for ($i = 0; $i < 1000; $i++) {
        // Get the space around the head.
        $space = $gamespace->exportSnakeSpace(4, 4);
        // Translate to flattened array.
        $arr = array();
        foreach ($space as $x => $y) {
            foreach ($y as $val) {
                $arr[] = $val;
            }
        }
        // Get the network to calculate the next move.
        $moves = $ontick($arr);
        foreach ($moves as $direction => $value) {
            if ($value > 0 && $gamespace->changeDirection($direction)) {
                break;
            }
        }

        $gamespace->tick();
    }

    return $gamespace->turns + ($gamespace->score * 2);
});

// Grab the best result.
$network = $trainer->bestNetwork();
$network->save(dirname(__FILE__) . '/../trained/game/snake.nn');
