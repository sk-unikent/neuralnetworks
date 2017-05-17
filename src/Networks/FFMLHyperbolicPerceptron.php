<?php
/**
 * Multi-layer Neural Network in PHP.
 *
 * @author S.Kelty <S.Kelty@kent.ac.uk>
 * @version 2.0
 * @license http://opensource.org/licenses/bsd-license.php BSD License
 */
namespace KeltyNN\Networks;

class FFMLHyperbolicPerceptron extends FFMLPerceptron
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

        $this->type = 'FFMLHyperbolicPerceptron';
        $this->version = '1.1';
    }

    /**
     * Implements the standard (default) activation function for backpropagation networks.
     *
     * @param float $value The preliminary output to apply this function to
     *
     * @return float The final output of the node
     */
    protected function activation($value)
    {
        return (exp($value) - exp(-$value)) / (exp($value) + exp(-$value));
    }
}
