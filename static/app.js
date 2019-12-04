const form = $(".my-form");
const checkbox = document.querySelector('.my-form input[type="checkbox"]');
const result = document.querySelector('.result');

checkbox.addEventListener("change", function() {
    const checked = this.checked;
    $("#is_exchangeable").val(checked ? 'True' : 'False');
});

form.on('submit', function (e) {
    e.preventDefault();

    var $loader = $("<div>", {'class': 'loading'});
    $(form).append($loader);

    $.ajax({
        type: 'POST',
        url: form.attr("action"),
        data: form.serialize(),
        success: function(res) {
            form.hide();
            $('#prediction').html(res.result + ' ' + res.currency);
            $(result).toggleClass('hidden');
            $loader.remove();
        },
        error: function () {
            alert('An error occurred.');
        }
    });
});

$('#go-back').on('click', function () {
    $(result).toggleClass('hidden');
    $('#prediction').html("");
    form.show();
});
