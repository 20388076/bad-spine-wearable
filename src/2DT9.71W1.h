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
                            if (x[35] <= 0.7885000109672546) {
                                return 1;
                            }

                            else {
                                if (x[31] <= 1.371999979019165) {
                                    if (x[19] <= -0.9149999916553497) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    if (x[34] <= 0.08950000256299973) {
                                        if (x[70] <= 0.07349999994039536) {
                                            return 1;
                                        }

                                        else {
                                            return 2;
                                        }
                                    }

                                    else {
                                        return 2;
                                    }
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