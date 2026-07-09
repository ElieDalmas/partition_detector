#pragma once


namespace omr {
    // Bounding box, [x1, y1, x2, y2)
    struct Box {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;

        int width() const { return x2 - x1; }
        int height() const { return y2 - y1; }
        double centerX() const { return (x1 + x2) / 2.0; }
        double centerY() const { return (y1 + y2) / 2.0; }
    };
}
