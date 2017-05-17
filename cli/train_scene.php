<?php
/**
 * Training script.
 *
 * A scene is like so:
 * (a, b, c, d, e, f)
 * Where:
 * a) Is a tree
 * b) Is a stream
 * c) Is a rock
 * d) Is a car
 * e) Is a skyscraper
 * f) Is a bus stop
 *
 * The point is to identify that if a car is present,
 * even if there are no skyscrapers or bus stops we can
 * infer that there could be in this setting.
 * Essentially I want the car neural path to semi-activate
 * the skyscraper and bus stop paths.
 *
 * This would likely be best served by a Hopfield net or similar.
 */

require_once(dirname(__FILE__) . '/../lib.php');

// TODO.
