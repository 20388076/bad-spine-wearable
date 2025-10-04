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
                        if (x[1] <= 0.016499999910593033) {
                            return 14;
                        }

                        else {
                            if (x[0] <= 1.0024999678134918) {
                                if (x[1] <= 0.027000000700354576) {
                                    if (x[2] <= 0.04800000041723251) {
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
                                                    if (x[2] <= 0.06499999947845936) {
                                                        return 6;
                                                    }

                                                    else {
                                                        if (x[8] <= 0.001500000071246177) {
                                                            if (x[2] <= 0.10450000315904617) {
                                                                return 6;
                                                            }

                                                            else {
                                                                if (x[2] <= 0.21199999749660492) {
                                                                    return 9;
                                                                }

                                                                else {
                                                                    if (x[2] <= 0.22449999302625656) {
                                                                        return 6;
                                                                    }

                                                                    else {
                                                                        return 9;
                                                                    }
                                                                }
                                                            }
                                                        }

                                                        else {
                                                            return 9;
                                                        }
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
                                    if (x[7] <= 0.0005000000237487257) {
                                        return 1;
                                    }

                                    else {
                                        if (x[3] <= 0.014500000048428774) {
                                            if (x[0] <= 0.9715000092983246) {
                                                return 13;
                                            }

                                            else {
                                                if (x[1] <= 0.02850000001490116) {
                                                    return 10;
                                                }

                                                else {
                                                    return 13;
                                                }
                                            }
                                        }

                                        else {
                                            if (x[2] <= 0.2474999949336052) {
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
                                if (x[1] <= 0.032500000670552254) {
                                    if (x[3] <= 0.012000000104308128) {
                                        return 5;
                                    }

                                    else {
                                        if (x[1] <= 0.02650000061839819) {
                                            if (x[2] <= 0.14650000631809235) {
                                                if (x[2] <= 0.07449999824166298) {
                                                    return 8;
                                                }

                                                else {
                                                    if (x[0] <= 1.0264999866485596) {
                                                        return 8;
                                                    }

                                                    else {
                                                        if (x[0] <= 1.0285000205039978) {
                                                            return 11;
                                                        }

                                                        else {
                                                            return 8;
                                                        }
                                                    }
                                                }
                                            }

                                            else {
                                                if (x[0] <= 1.0425000190734863) {
                                                    if (x[2] <= 0.19349999725818634) {
                                                        if (x[0] <= 1.0364999771118164) {
                                                            return 11;
                                                        }

                                                        else {
                                                            if (x[0] <= 1.0379999876022339) {
                                                                return 8;
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
                                                    return 8;
                                                }
                                            }
                                        }

                                        else {
                                            return 11;
                                        }
                                    }
                                }

                                else {
                                    return 2;
                                }
                            }
                        }
                    }

                protected:
                };
            }
        }
    }