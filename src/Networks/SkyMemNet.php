<?php
namespace KeltyNN\Networks;

/**
 * A memory network, capable of storing and remembering data.
 * @todo Memory weights
 * @todo Memory LRU/MRU
 */
class SkyMemNet extends \KeltyNN\NeuralNetwork
{
    // Keep it compatible with the FFML perceptrons.
    public $nodeThreshold = array(1 => array());
    public $edgeWeights = array(1 => array());

    /**
     * Constructor.
     */
    public function __construct() {
       $this->type = 'SkyMemNet';
       $this->version = '1.0';
    }

    /**
     * Create a new object.
     */
    protected function create($value) {
        $hash = hash('sha512', $value);
        if (!isset($this->nodeThreshold[1][$hash])) {
            $this->nodeThreshold[1][$hash] = 1;
            $this->edgeWeights[1][$hash] = array();
        }

        return $hash;
    }

    /**
     * Create a link between two objects if it doesnt exist already.
     * We can only strengthen links, never decrement them.
     */
    protected function link($node, $related, $strength = 1) {
        if (!isset($this->edgeWeights[1][$node][$related]) || $this->edgeWeights[1][$node][$related] < $strength) {
            $this->edgeWeights[1][$node][$related] = $strength;
        }
    }

    /**
     * Remember this!
     */
    public function store($value, $related = array()) {
        $node = $this->create($value);
        foreach ($related as $relative => $strength) {
            $relative = $this->create($relative);
            $this->link($node, $relative, $strength);
            $this->link($relative, $node, $strength);
        }
    }
}
