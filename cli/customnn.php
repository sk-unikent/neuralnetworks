<?php
/**
 * Manually create an optimised XOR.
 */

require_once(dirname(__FILE__) . '/../lib.php');

$data = file_get_contents(dirname(__FILE__) . "/../trained/maths/basic/xor.nn");
$data = unserialize($data);
$data['type'] = 'FFMLLinearPerceptron';
$data['version'] = '1.1';
$data['nodeCount'] = array(2, 1, 1);
$data['edgeWeight'] = array(
    0 => array(
        0 => array(
            0 => 1,
            1 => 1
        ),
        1 => array(
            0 => 1,
            1 => 1
        )
    ),
    1 => array(
        0 => array(
            0 => 1
        )
    ),
);
$data['nodeThreshold'] = array(
    1 => array(
        0 => 0
    ),
    2 => array(
        0 => 0
    )
);

$newnn = KeltyNN\NeuralNetwork::load($data);

// Add test-data to the network.
for ($i = 0; $i < 1000; $i++) {
    $a = rand(0, 900);
    $b = rand(0, 900);
    $newnn->addTestData(array($a, $b), array($a + $b));
}

$max = 10;
$j = 0;

// Train the network.
while (!($success = $newnn->train(10000, 0.1)) && ++$j < $max) {
}

$result = $newnn->calculate(array(56, 4));
echo "56 + 4 = {$result[0]}\n";

// print a message if the network was succesfully trained
if ($success) {
    $epochs = $newnn->getEpoch();
    $newnn->save(dirname(__FILE__) . '/../trained/maths/basic/addition.nn');
    echo "Success in $epochs training rounds!\n";
    exit(0);
}
