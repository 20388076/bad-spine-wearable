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
                        if (x[69] <= 7.150000095367432) {
                            if (x[14] <= -0.5794999897480011) {
                                return 1;
                            }

                            else {
                                if (x[17] <= 0.0409999992698431) {
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