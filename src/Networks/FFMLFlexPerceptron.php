<?php
/**
 * @author Skylar Kelty
 * @license http://opensource.org/licenses/bsd-license.php BSD License
 */
namespace KeltyNN\Networks;

class FFMLFlexPerceptron extends FFMLPerceptron
{
    protected $designed = false;
    public $layerConnectors = array();

    /**
     * Creates a neural network.
     *
     * Example:
     * <code>
     * // create a network with 4 input nodes, 10 hidden nodes, and 4 output nodes
     * $n = new NeuralNetwork(4, 10, 4);
     *
     * // create a network with 4 input nodes, 1 hidden layer with 10 nodes,
     * // another hidden layer with 10 nodes, and 4 output nodes
     * $n = new NeuralNetwork(4, 10, 10, 4);
     *
     * // alternative syntax
     * $n = new NeuralNetwork(array(4, 10, 10, 4));
     * </code>
     *
     * @param array $nodeCount The number of nodes in the consecutive layers.
     */
    public function __construct($nodeCount) {
        if (!is_array($nodeCount)) {
            $nodeCount = func_get_args();
        }
        parent::__construct($nodeCount);

        $this->type = 'FFMLFlexPerceptron';
        $this->version = '1.1';
    }

    /**
     * Enable designer mode.
     */
    public function setDesignerMode() {
        $this->designed = true;
    }

    /**
     * Create a blank node with a random, or given weight.
     */
    public function addNode($layer, $weight = false) {
        if ($weight == false) {
            $weight = $this->getRandomWeight($weight);
        }

        if ($layer == count($this->nodeThreshold) - 1) {
            $layer = count($this->nodeCount) - 1;
            $this->nodeThreshold[$layer + 1] = $this->nodeThreshold[$layer];
            $this->nodeCount[$layer + 1] = $this->nodeCount[$layer];
            $this->nodeThreshold[$layer] = array();
            $this->nodeCount[$layer] = 1;
            $this->layerCount++;
        } else {
            $this->nodeCount[$layer]++;
        }

        $this->nodeThreshold[$layer][] = $weight;
        return count($this->nodeThreshold[$layer]) - 1;
    }

    /**
     * Connect two nodes together.
     */
    public function connect($layer, $node, $tolayer, $tonode, $weight = false) {
        if ($weight == false) {
            $weight = $this->getRandomWeight($weight);
        }

        if (!isset($this->edgeWeight[$layer])) {
            $this->edgeWeight[$layer] = array();
        }

        if (!isset($this->edgeWeight[$layer][$node])) {
            $this->edgeWeight[$layer][$node] = array();
        }

        $this->edgeWeight[$layer][$node]["{$tolayer}_{$tonode}"] = $weight;
        $this->notifyConnection($layer, $node, $tolayer, $tonode);
    }

    /**
     * Return a random node in this network.
     */
    public function getRandomNode($layers) {
        if (empty($layers)) {
            return null;
        }

        // Make sure we can find a valid node.
        $valid = array();
        foreach ($layers as $layer) {
            if (isset($this->nodeCount[$layer]) && $this->nodeCount[$layer] > 0) {
                $valid[$layer] = $layer;
                break;
            }
        }

        if (empty($valid)) {
            return null;
        }

        // Get a random node.
        $layer = array_rand($valid);
        $node = rand(0, $this->nodeCount[$layer] - 1);
        $weight = 0;
        if (isset($this->nodeValue[$layer][$node])) {
            $weight = $this->nodeValue[$layer][$node];
        }

        return array(
            'layer' => $layer,
            'node' => $node,
            'weight' => $weight
        );
    }

    /**
     * Return a random edge in this network.
     */
    public function getRandomEdge() {
        $layer = array_rand($this->edgeWeight);
        $node = array_rand($this->edgeWeight[$layer]);
        $to = array_rand($this->edgeWeight[$layer][$node]);
        $weight = $this->edgeWeight[$layer][$node][$to];

        return array(
            'layer' => $layer,
            'node' => $node,
            'to' => $to,
            'weight' => $weight
        );
    }

    /**
     * Set the weight of a node.
     */
    public function setNodeWeight($layer, $node, $weight) {
        $this->nodeValue[$layer][$node] = $weight;
    }

    /**
     * Set the weight of an edge.
     */
    public function setEdgeWeight($layer, $node, $to, $weight) {
        $this->edgeWeight[$layer][$node][$to] = $weight;
    }

    /**
     * Upgrade a standard perceptron to a flex perceptron.
     */
    public static function upgrade($data) {
        $data['type'] = 'FFMLFlexPerceptron';
        $data['version'] = '1.1';
        $data['designed'] = true;

        // Upgrade edge weights to new format.
        $newedgeWeight = array();
        $layerConnectors = array();
        foreach ($data['edgeWeight'] as $layer => $nodes) {
            $newedgeWeight[$layer] = array();
            $tolayer = $layer++;
            foreach ($nodes as $node => $links) {
                $newedgeWeight[$layer][$node] = array();
                foreach ($links as $tonode => $weight) {
                    $newedgeWeight[$layer][$node]["{$tolayer}_{$tonode}"] = $weight;

                    if (!isset($layerConnectors["{$tolayer}_{$tonode}"])) {
                        $layerConnectors["{$tolayer}_{$tonode}"] = array();
                    }

                    if (!isset($layerConnectors["{$tolayer}_{$tonode}"][$layer])) {
                        $layerConnectors["{$tolayer}_{$tonode}"][$layer] = array();
                    }

                    if (!in_array($node, $layerConnectors["{$tolayer}_{$tonode}"][$layer])) {
                        $layerConnectors["{$tolayer}_{$tonode}"][$layer][] = $node;
                    }
                }
            }
        }

        $data['edgeWeight'] = $newedgeWeight;
        $data['layerConnectors'] = $layerConnectors;

        return $data;
    }

    /**
     * Exports the neural network.
     *
     * @returns array
     */
    public function export() {
        $store = parent::export();
        $store['layerConnectors'] = $this->layerConnectors;
        $store['designed'] = $this->designed;
        return $store;
    }

    /**
     * Calculate the output of the neural network for a given input vector.
     *
     * @param array $input The vector to calculate
     *
     * @return mixed The output of the network
     */
    public function calculate($input) {
        // put the input vector on the input nodes
        foreach ($input as $index => $value) {
            $this->nodeValue[0][$index] = $value;
        }

        // iterate the hidden layers
        for ($layer = 1; $layer < $this->layerCount; $layer++) {
            // iterate each node in this layer
            for ($node = 0; $node < ($this->nodeCount[$layer]); $node++) {
                $node_value = 0.0;

                // Any node in any layer might have a connection to this node
                // on basis of this, calculate this node's value
                $str = "{$layer}_{$node}";
                if (isset($this->layerConnectors[$str])) {
                    foreach ($this->layerConnectors[$str] as $ilayer => $inodes) {
                        foreach ($inodes as $inode) {
                            if (!isset($this->nodeValue[$ilayer]) || !isset($this->nodeValue[$ilayer][$inode])) {
                                print_r($this->layerConnectors);
                                print_r($this->nodeCount);
                                echo $ilayer . ' / ' . $inode . ' / ' . $this->layerCount . ' / '. $this->nodeCount[$layer];
                                die();
                            }
                            $inputnode_value = $this->nodeValue[$ilayer][$inode];
                            $edge_weight = $this->edgeWeight[$ilayer][$inode][$str];

                            $node_value = $node_value + ($inputnode_value * $edge_weight);
                        }
                    }
                }

                // apply the threshold
                if (!isset($this->nodeThreshold[$layer]) || !isset($this->nodeThreshold[$layer][$node])) {
                    echo $layer . ' / ' . $node;
                    print_r($this->nodeThreshold);
                    die();
                }
                $node_value = $node_value - $this->nodeThreshold[$layer][$node];

                // apply the activation function
                $node_value = $this->activation($node_value);

                // remember the outcome
                $this->nodeValue[$layer][$node] = $node_value;
            }
        }

        // return the values of the last layer (the output layer)
        return $this->nodeValue[$this->layerCount - 1];
    }

    /**
     * Notify the layerConnectors array of a new connection.
     */
    protected function notifyConnection($layer, $node, $tolayer, $tonode) {
        if (!isset($this->layerConnectors["{$tolayer}_{$tonode}"])) {
            $this->layerConnectors["{$tolayer}_{$tonode}"] = array();
        }

        if (!isset($this->layerConnectors["{$tolayer}_{$tonode}"][$layer])) {
            $this->layerConnectors["{$tolayer}_{$tonode}"][$layer] = array();
        }

        if (!in_array($node, $this->layerConnectors["{$tolayer}_{$tonode}"][$layer])) {
            $this->layerConnectors["{$tolayer}_{$tonode}"][$layer][] = $node;
        }
    }

    /**
     * Get all backward connections for this node.
     */
    protected function getPrevLayer($layer, $node) {
        if (isset($this->layerConnectors["{$layer}_{$node}"])) {
            return $this->layerConnectors["{$layer}_{$node}"];
        }

        return array();
    }

    /**
     * Randomise the weights in the neural network.
     */
    protected function initWeights() {
        // assign a random value to each edge between the layers, and randomise each threshold
        //
        // 1. start at layer '1' (so skip the input layer)
        for ($layer = 1; $layer < $this->layerCount; $layer++) {
            $prev_layer = $layer - 1;

            // 2. in this layer, walk each node
            for ($node = 0; $node < $this->nodeCount[$layer]; $node++) {

                // 3. randomise this node's threshold
                $this->nodeThreshold[$layer][$node] = $this->getRandomWeight($layer);

                // If we already have structure we dont just want to blindly link previous layers.
                if (!$this->designed) {
                    // 4. this node is connected to each node of the previous layer
                    for ($prev_index = 0; $prev_index < $this->nodeCount[$prev_layer]; $prev_index++) {

                        // 5. this is the 'edge' that needs to be reset / initialised
                        $this->edgeWeight[$prev_layer][$prev_index]["{$layer}_{$node}"] = $this->getRandomWeight($prev_layer);
                        $this->notifyConnection($prev_layer, $prev_index, $layer, $node);

                        // 6. initialize the 'previous weightcorrection' at 0.0
                        $this->previousWeightCorrection[$prev_layer][$prev_index] = 0.0;
                    }
                } else {
                    // 5. This node is connected to random nodes.
                    $connected = $this->getPrevLayer($layer, $node);
                    foreach ($connected as $ilayer => $inodes) {
                        foreach ($inodes as $inode) {
                            // This is the 'edge' that needs to be reset / initialised
                            $this->edgeWeight[$ilayer][$inode]["{$layer}_{$node}"] = $this->getRandomWeight($ilayer);
                            $this->notifyConnection($ilayer, $inode, $layer, $node);

                            // Initialize the 'previous weightcorrection' at 0.0
                            $this->previousWeightCorrection[$ilayer][$inode] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /**
     * Performs the backpropagation algorithm. This changes the weights and thresholds of the network.
     *
     * @param array $output         The output obtained by the network
     * @param array $desired_output The desired output
     */
    protected function backpropagate($output, $desired_output) {
        $errorgradient = array();
        $outputlayer = $this->layerCount - 1;

        $momentum = $this->getMomentum();

        // Propagate the difference between output and desired output through the layers.
        for ($layer = $this->layerCount - 1; $layer > 0; $layer--) {
            for ($node = 0; $node < $this->nodeCount[$layer]; $node++) {

                // step 1: determine errorgradient
                if ($layer == $outputlayer) {
                    // for the output layer:
                    // 1a. calculate error between desired output and actual output
                    $error = $desired_output[$node] - $output[$node];

                    // 1b. calculate errorgradient
                    $errorgradient[$layer][$node] = $this->derivativeActivation($output[$node]) * $error;
                } else {
                    // for hidden layers:
                    // 1a. sum the product of edgeWeight and errorgradient of the 'next' layer
                    $productsum = 0;
                    foreach ($this->edgeWeight[$layer][$node] as $next_ident => $_edgeWeight) {
                        list($next_layer, $next_index) = explode('_', $next_ident);
                        $_errorgradient = $errorgradient[$next_layer][$next_index];
                        $productsum = $productsum + $_errorgradient * $_edgeWeight;
                    }

                    // 1b. calculate errorgradient
                    $nodeValue = $this->nodeValue[$layer][$node];
                    $errorgradient[$layer][$node] = $this->derivativeActivation($nodeValue) * $productsum;
                }

                // step 2: use the errorgradient to determine a weight correction for each node
                $prev_layer = $layer - 1;
                $learning_rate = $this->getlearningRate($prev_layer);

                $connected = $this->getPrevLayer($layer, $node);
                foreach ($connected as $prev_layer => $prev_indexes) {
                    foreach ($prev_indexes as $prev_index) {
                        // 2a. obtain nodeValue, edgeWeight and learning rate
                        $nodeValue = $this->nodeValue[$prev_layer][$prev_index];
                        $edgeWeight = $this->edgeWeight[$prev_layer][$prev_index]["{$layer}_{$node}"];

                        // 2b. calculate weight correction
                        $weight_correction = $learning_rate * $nodeValue * $errorgradient[$layer][$node];

                        // 2c. retrieve previous weight correction
                        $prev_weightcorrection = @$this->previousWeightCorrection[$layer][$node];

                        // 2d. combine those ('momentum learning') to a new weight
                        $new_weight = $edgeWeight + $weight_correction + $momentum * $prev_weightcorrection;

                        // 2e. assign the new weight to this edge
                        $this->edgeWeight[$prev_layer][$prev_index]["{$layer}_{$node}"] = $new_weight;
                        $this->notifyConnection($prev_layer, $prev_index, $layer, $node);

                        // 2f. remember this weightcorrection
                        $this->previousWeightCorrection[$layer][$node] = $weight_correction;
                    }
                }

                // step 3: use the errorgradient to determine threshold correction
                $threshold_correction = $learning_rate * -1 * $errorgradient[$layer][$node];
                $new_threshold = $this->nodeThreshold[$layer][$node] + $threshold_correction;

                $this->nodeThreshold[$layer][$node] = $new_threshold;
            }
        }
    }
}
