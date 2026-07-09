#include "omr/knn.h"
#include "test_utils.h"

int main() {
    std::vector<std::vector<double> > X = {
        {0.0, 0.0}, {0.1, 0.1}, {-0.1, 0.1},
        {10.0, 10.0}, {10.1, 9.9}, {9.9, 10.1},
    };
    std::vector<std::string> y = {"a", "a", "a", "b", "b", "b"};

    omr::KNeighborsClassifier knn(1);
    knn.fit(X, y);

    auto preds = knn.predict({{0.0, 0.05}, {10.0, 10.0}});
    //check predict classes
    CHECK(preds[0] == "a");
    CHECK(preds[1] == "b");

    // check predict proba
    auto proba = knn.predictProba({{0.0, 0.05}});
    double sum = 0.0;
    for (double p: proba[0])
        sum += p;
    CHECK(equal(sum, 1.0));

    return 0;
}
