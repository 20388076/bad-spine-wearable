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
                        if (x[27] <= 0.007500000298023224) {
                            return 4;
                        }

                        else {
                            if (x[31] <= 1.2294999957084656) {
                                if (x[14] <= -0.12150000035762787) {
                                    if (x[3] <= 0.013500000350177288) {
                                        return 1;
                                    }

                                    else {
                                        return 0;
                                    }
                                }

                                else {
                                    if (x[10] <= 1.562000036239624) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }
                            }

                            else {
                                if (x[24] <= -0.010499999858438969) {
                                    if (x[28] <= -0.022499999962747097) {
                                        return 2;
                                    }

                                    else {
                                        return 3;
                                    }
                                }

                                else {
                                    return 3;
                                }
                            }
                        }
                    }

                protected:
                };
            }
        }
    }