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
                        if (x[69] <= 0.8948870301246643) {
                            if (x[35] <= 0.5170800685882568) {
                                return 1;
                            }

                            else {
                                if (x[31] <= -1.2919584512710571) {
                                    if (x[19] <= -1.035331130027771) {
                                        return 2;
                                    }

                                    else {
                                        return 1;
                                    }
                                }

                                else {
                                    if (x[34] <= -1.176056146621704) {
                                        return 1;
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