<?php

namespace KeltyNN\Input;

class Image
{
    /**
     * Serialize a given image for a neural net.
     *
     * @param [type] $filename [description]
     *
     * @return [type] [description]
     */
    public static function serialize($filename, $network, $greyscale = false, $blackwhite = false)
    {
        $max = $network->getInputCount();

        $image = new \Imagick($filename);
        if ($image->getImageColorspace() != \Imagick::COLORSPACE_SRGB) {
            $image->transformimagecolorspace(\Imagick::COLORSPACE_SRGB);
        }

        $w = $image->getImageWidth();
        $h = $image->getImageHeight();

        // Rescale the image until we will fit into input neuron space.
        $perpixel = $blackwhite ? 1 : 3;
        while ($w * $h * $perpixel > $max) {
            $aspect = $w / $h;
            $w = $w - 10;
            $h = $h - floor(10 * $a);
        }

        $image->adaptiveResizeImage($w, $h);

        // Build a flattened RGB array.
        $flattened = array();

        $it = $image->getPixelIterator();
        foreach ($it as $row => $pixels) {
            foreach ($pixels as $column => $pixel) {
                $rgb = $pixel->getColor();

                if ($blackwhite) {
                    $flattened[] = $rgb[0] == 255 ? 1 : 0;
                } else {
                    $flattened[] = $rgb[0];
                    $flattened[] = $rgb[1];
                    $flattened[] = $rgb[2];
                }
            }
        }

        return $flattened;
    }
}
