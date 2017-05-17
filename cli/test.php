<?php
/**
 * Testing script.
 */
require_once(dirname(__FILE__) . '/../lib.php');

// Create a Perceptron network.
$eq = KeltyNN\NeuralNetwork::loadfile(dirname(__FILE__) . '/../trained/maths/advanced/equals.nn');

// Add test-data to the network.
$testdata = array();
for ($i = 0; $i < 100; $i++) {
    $a = rand(1, 500);
    $b = rand(1, 500);
    if (rand(1, 2) == 1) {
        $testdata[] = array(array($a, $a), array(1));
    } else {
        $testdata[] = array(array($a, $b), array($a == $b ? 1 : -1));
    }
}

for ($i = 0; $i < 100; $i++) {
    $a = rand(1, 500);
    $b = rand(1, 500);
    if (rand(1, 2) == 1) {
        $b = $a;
    }

    $result = $eq->calculate(array($a, $b));
    $result = round($result[0]) == 1 ? 1 : -1;

    echo "{$a} == {$b} = {$result}\n";
}
