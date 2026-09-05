module DCMDynamicPPLExt

import DeepCompartmentModels as DCM
import DynamicPPL
import Random

function DCM._validate_individual_model(
        model::DynamicPPL.Model, locals::DCM.LocalVariables, values)
    vi = DynamicPPL.VarInfo(
        Random.Xoshiro(0), model, DynamicPPL.InitFromParams(values, nothing))
    collect(keys(vi)) == [DynamicPPL.VarName{locals.site}()] ||
        throw(ArgumentError("individual model must sample only the declared local site"))
    return nothing
end

DCM._individual_logjoint(model::DynamicPPL.Model, values) =
    DynamicPPL.logjoint(model, values)

DCM._individual_loglikelihood(model::DynamicPPL.Model, values) =
    DynamicPPL.loglikelihood(model, values)

end
