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
                            if (x[10] <= 2.191499948501587) {
                                return 2;
                            }

                            else {
                                if (x[11] <= 1.2985000014305115) {
                                    if (x[33] <= 0.625) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    return 1;
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