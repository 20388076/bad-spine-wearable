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
                        if (x[27] <= 0.034374202601611614) {
                            return 4;
                        }

                        else {
                            if (x[31] <= 0.15674462169408798) {
                                if (x[14] <= 0.4247552901506424) {
                                    if (x[3] <= 0.0904422327876091) {
                                        return 1;
                                    }

                                    else {
                                        return 0;
                                    }
                                }

                                else {
                                    if (x[10] <= -0.6980878114700317) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }
                            }

                            else {
                                if (x[24] <= 0.10565854609012604) {
                                    if (x[28] <= -0.6073848307132721) {
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