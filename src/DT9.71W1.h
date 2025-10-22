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
                        if (x[69] <= 0.8534342050552368) {
                            if (x[34] <= 0.2513910084962845) {
                                if (x[21] <= -0.7566248774528503) {
                                    return 2;
                                }

                                else {
                                    return 1;
                                }
                            }

                            else {
                                if (x[15] <= 0.34438471496105194) {
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