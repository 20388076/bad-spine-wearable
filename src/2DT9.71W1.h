#pragma once
#include <cstdarg>
namespace Eloquent {
    namespace ML {
        namespace Port {
            class DecisionTree {
                public:
                    /**
                    * Predict class for features vector
                    */
                    int predict(float *x) {
                        if (x[69] <= 687.4985046386719) {
                            if (x[34] <= 2.325000047683716) {
                                if (x[21] <= -7.864999771118164) {
                                    return 2;
                                }

                                else {
                                    return 1;
                                }
                            }

                            else {
                                if (x[15] <= -6.141000032424927) {
                                    return 1;
                                }

                                else {
                                    return 2;
                                }
                            }
                        }

                        else {
                            return 0;
                        }
                    }

                protected:
                };
            }
        }
    }