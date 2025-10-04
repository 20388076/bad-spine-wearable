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
                        if (x[0] <= 0.9914999902248383) {
                            if (x[1] <= 0.027000000700354576) {
                                if (x[0] <= 0.9814999997615814) {
                                    if (x[4] <= 0.0005000000237487257) {
                                        if (x[0] <= 0.9805000126361847) {
                                            return 6;
                                        }

                                        else {
                                            if (x[8] <= 0.0005000000237487257) {
                                                return 6;
                                            }

                                            else {
                                                if (x[2] <= 0.09549999982118607) {
                                                    if (x[1] <= 0.024500000290572643) {
                                                        return 9;
                                                    }

                                                    else {
                                                        return 6;
                                                    }
                                                }

                                                else {
                                                    return 9;
                                                }
                                            }
                                        }
                                    }

                                    else {
                                        return 3;
                                    }
                                }

                                else {
                                    if (x[2] <= 0.06399999931454659) {
                                        return 12;
                                    }

                                    else {
                                        if (x[8] <= 0.0010000000474974513) {
                                            return 0;
                                        }

                                        else {
                                            return 9;
                                        }
                                    }
                                }
                            }

                            else {
                                if (x[2] <= 0.16499999910593033) {
                                    return 1;
                                }

                                else {
                                    if (x[3] <= 0.014500000048428774) {
                                        if (x[0] <= 0.9715000092983246) {
                                            if (x[0] <= 0.9705000221729279) {
                                                return 13;
                                            }

                                            else {
                                                return 13;
                                            }
                                        }

                                        else {
                                            if (x[1] <= 0.02850000001490116) {
                                                return 10;
                                            }

                                            else {
                                                return 10;
                                            }
                                        }
                                    }

                                    else {
                                        if (x[2] <= 0.2485000044107437) {
                                            if (x[2] <= 0.22949999570846558) {
                                                return 7;
                                            }

                                            else {
                                                if (x[0] <= 0.9805000126361847) {
                                                    return 4;
                                                }

                                                else {
                                                    return 4;
                                                }
                                            }
                                        }

                                        else {
                                            return 7;
                                        }
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
                                        if (x[8] <= 0.0005000000237487257) {
                                            return 8;
                                        }

                                        else {
                                            if (x[1] <= 0.02650000061839819) {
                                                if (x[0] <= 1.0224999785423279) {
                                                    return 8;
                                                }

                                                else {
                                                    if (x[8] <= 0.001500000071246177) {
                                                        if (x[0] <= 1.0425000190734863) {
                                                            if (x[3] <= 0.014500000048428774) {
                                                                if (x[0] <= 1.0394999980926514) {
                                                                    return 11;
                                                                }

                                                                else {
                                                                    return 11;
                                                                }
                                                            }

                                                            else {
                                                                if (x[2] <= 0.09449999779462814) {
                                                                    return 11;
                                                                }

                                                                else {
                                                                    if (x[2] <= 0.11549999937415123) {
                                                                        return 8;
                                                                    }

                                                                    else {
                                                                        return 11;
                                                                    }
                                                                }
                                                            }
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