<?php
/**
 * Benchmarking script.
 */
require_once(dirname(__FILE__) . '/../lib.php');

// Create a Perceptron network.
$xor = KeltyNN\NeuralNetwork::loadfile(dirname(__FILE__) . '/../trained/maths/basic/xor.nn');

// Create a Perceptron network.
$flex_xor = KeltyNN\NeuralNetwork::loadfile(dirname(__FILE__) . '/../trained/maths/basic/flex_xor.nn');

foreach (array($xor, $flex_xor) as $network) {
    $time = microtime(true);
    for ($i = 0; $i < 100000; $i++) {
        $network->calculate(array(1, 1));
        $network->calculate(array(1, 0));
        $network->calculate(array(0, 1));
        $network->calculate(array(0, 0));
    }

    echo $network->getTitle() . ' finished in ' . (microtime(true) - $time) . "s.\n";
}
