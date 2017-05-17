<?php
/**
 * Leaky RELU network.
 *
 * @license http://opensource.org/licenses/bsd-license.php BSD License
 */
namespace KeltyNN\Networks;

class FFMLLeakyRELUPerceptron extends FFMLPerceptron
{
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
    public function __construct($nodeCount)
    {
        if (!is_array($nodeCount)) {
            $nodeCount = func_get_args();
        }
        parent::__construct($nodeCount);

        $this->type = 'FFMLLeakyRELUPerceptron';
        $this->version = '1.1';
    }

    /**
     * Implements a simple linear function.
     *
     * @param float $value The preliminary output to apply this function to
     *
     * @return float The final output of the node
     */
    public function activation($value)
    {
        return ($value >= 0) ? $value : 0.01 * $value;
    }

    /**
     * Implements the derivative of the activation function. By default, this is the
     * inverse of the 'tanh' activation function: 1.0 - tanh($value)*tanh($value);.
     *
     * @param float $value 'X'
     *
     * @return $float
     */
    public function derivativeActivation($value)
    {
        return ($value >= 0) ?  1.0 : 0.01 * $value;
    }
}
