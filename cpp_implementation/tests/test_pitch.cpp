#include "omr/pitch.h"
#include "test_utils.h"

int main() {
    auto rel = omr::parseRelPosition("instance:#000042;duration:8;rel_position:-4;");
    CHECK(rel.has_value());
    CHECK(*rel == -4);

    CHECK(!omr::parseRelPosition("instance:#000042;duration:8;").has_value());

    // 5 lines staff, spacing 10
    std::vector staffYs = {10, 20, 30, 40, 50};

    // check pitch high
    CHECK(omr::relPositionFromY(30.0, staffYs) == 0); // middle line
    CHECK(omr::relPositionFromY(10.0, staffYs) == 4); // top line
    CHECK(omr::relPositionFromY(50.0, staffYs) == -4); // bottom line
    CHECK(omr::relPositionFromY(15.0, staffYs) == 3); // space between top two lines

    return 0;
}
