--
-- Created by IntelliJ IDEA.
-- User: rgobbel
-- Date: 4/20/16
-- Time: 8:10 PM
-- To change this template use File | Settings | File Templates.
--

function map(func, array)
    local new_array = {}
    for i, v in ipairs(array) do
        new_array[#new_array + 1] = func(v)
    end
    return new_array
end

function mapn(func, ...)
    -- Variadic arguments bound to an array.
    local arrays = {...}
    local new_array = {}
    -- Simple for-loop.
    local i = 1
    while true do
        local arg_list = map(function(arr) return arr[i] end, arrays)
        if #arg_list == 0 then
            break
        end
        new_array[i] = func(unpack(arg_list))
        i = i + 1
    end
    return new_array
end


-- Using 'mapn' instead of 'map' (probably how you intended).
function zip(...)
    return mapn(function(...) return {...} end,...)
end

-- Same as before.
function transpose(...)
    return zip(unpack(...))
end


