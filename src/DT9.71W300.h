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
                        if (x[0] <= 0.9919999837875366) {
                            if (x[3] <= 0.014500000048428774) {
                                if (x[1] <= 0.027000000700354576) {
                                    if (x[2] <= 0.04899999871850014) {
                                        return 3;
                                    }

                                    else {
                                        if (x[8] <= 0.0005000000237487257) {
                                            return 0;
                                        }

                                        else {
                                            if (x[0] <= 0.9814999997615814) {
                                                if (x[0] <= 0.9805000126361847) {
                                                    return 6;
                                                }

                                                else {
                                                    if (x[2] <= 0.0885000005364418) {
                                                        return 6;
                                                    }

                                                    else {
                                                        return 9;
                                                    }
                                                }
                                            }

                                            else {
                                                if (x[8] <= 0.0025000000605359674) {
                                                    return 9;
                                                }

                                                else {
                                                    return 12;
                                                }
                                            }
                                        }
                                    }
                                }

                                else {
                                    if (x[0] <= 0.9715000092983246) {
                                        return 13;
                                    }

                                    else {
                                        if (x[2] <= 0.2654999941587448) {
                                            return 10;
                                        }

                                        else {
                                            return 10;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[2] <= 0.1744999960064888) {
                                    return 1;
                                }

                                else {
                                    if (x[2] <= 0.24800000339746475) {
                                        return 4;
                                    }

                                    else {
                                        return 7;
                                    }
                                }
                            }
                        }

                        else {
                            if (x[3] <= 0.00849999999627471) {
                                return 14;
                            }

                            else {
                                if (x[3] <= 0.012000000104308128) {
                                    return 5;
                                }

                                else {
                                    if (x[1] <= 0.032500000670552254) {
                                        if (x[1] <= 0.02650000061839819) {
                                            if (x[2] <= 0.13750000298023224) {
                                                if (x[2] <= 0.09750000014901161) {
                                                    return 8;
                                                }

                                                else {
                                                    return 8;
                                                }
                                            }

                                            else {
                                                if (x[0] <= 1.0415000319480896) {
                                                    if (x[8] <= 0.001500000071246177) {
                                                        return 11;
                                                    }

                                                    else {
                                                        return 8;
                                                    }
                                                }

                                                else {
                                                    return 8;
                                                }
                                            }
                                        }

                                        else {
                                            return 11;
                                        }
                                    }

                                    else {
                                        return 2;
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