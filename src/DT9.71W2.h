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
                        if (x[10] <= 0.5655000135302544) {
                            return 14;
                        }

                        else {
                            if (x[71] <= 72.39599800109863) {
                                return 2;
                            }

                            else {
                                if (x[20] <= -0.22749999165534973) {
                                    if (x[29] <= -0.019999999552965164) {
                                        return 1;
                                    }

                                    else {
                                        if (x[38] <= 0.020999999716877937) {
                                            if (x[30] <= -0.019000000320374966) {
                                                if (x[21] <= -0.9845000207424164) {
                                                    if (x[12] <= 3.1260000467300415) {
                                                        return 12;
                                                    }

                                                    else {
                                                        return 3;
                                                    }
                                                }

                                                else {
                                                    if (x[73] <= 0.6000000238418579) {
                                                        return 6;
                                                    }

                                                    else {
                                                        if (x[47] <= 0.00849999999627471) {
                                                            return 9;
                                                        }

                                                        else {
                                                            return 6;
                                                        }
                                                    }
                                                }
                                            }

                                            else {
                                                return 0;
                                            }
                                        }

                                        else {
                                            if (x[3] <= 0.014500000048428774) {
                                                if (x[20] <= -0.9195000231266022) {
                                                    return 13;
                                                }

                                                else {
                                                    return 10;
                                                }
                                            }

                                            else {
                                                if (x[31] <= 1.2139999866485596) {
                                                    return 4;
                                                }

                                                else {
                                                    return 7;
                                                }
                                            }
                                        }
                                    }
                                }

                                else {
                                    if (x[72] <= 0.15300000086426735) {
                                        return 5;
                                    }

                                    else {
                                        if (x[59] <= 0.030499999411404133) {
                                            return 8;
                                        }

                                        else {
                                            if (x[13] <= -0.988999992609024) {
                                                return 8;
                                            }

                                            else {
                                                return 11;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                protected:
                };
            }
        }
    }